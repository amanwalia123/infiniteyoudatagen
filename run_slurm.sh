#!/usr/bin/env bash
#
# run_slurm.sh — Download models + generate data on a Slurm cluster.
#
# Steps:
#   1. Create/activate a Python virtual environment
#   2. Install requirements
#   3. Download all required models on node 0 (skips if already present)
#   4. Run data generation across all GPUs on every node
#
# Usage:
#   # Edit the SBATCH directives below, then:
#   CLUSTER_LAYOUT=4,4,2 sbatch run_slurm.sh
#
# Environment variables (all optional except CLUSTER_LAYOUT):
#
#   CLUSTER_LAYOUT     Comma-separated GPU counts per node (REQUIRED, e.g. 4,4,2)
#   MODEL_DIR          Where to download / find models     (default: ./models)
#   CELEB_HQ_ROOT      CelebHQ image directory             (default: ./data/CelebHQRefForRelease)
#   GENDER_METADATA    Gender metadata JSON                 (default: gender_map.json)
#   MODEL_VERSION      sim_stage1 | aes_stage2              (default: sim_stage1)
#   SCENE_PACKS_FILE   Scene packs JSON                     (default: scene_packs/scene_packs_large1.json)
#   PROMPT_FILE        CSV with prompt/identity/file_id     (default: splits/train_metadata.csv)
#   NUM_SAMPLES        Samples per GPU                      (default: 500)
#   NUM_REPEAT         Repeats per prompt                   (default: 3)
#   OUTPUT_DIR         Output directory                     (default: let data_generator.py decide)
#   SKIP_DOWNLOAD      Set to 1 to skip model download      (default: 0)
#   HF_TOKEN           HuggingFace token for gated models   (default: from env or prompted)
#

#SBATCH --job-name=infu_gen
#SBATCH --output=logs/%x_%j_node%N.out
#SBATCH --error=logs/%x_%j_node%N.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
VENV_DIR=".venv"
: "${CLUSTER_LAYOUT:?Must set CLUSTER_LAYOUT, e.g. 4,4,2}"

MODEL_DIR="${MODEL_DIR:-./models}"
CELEB_HQ_ROOT="${CELEB_HQ_ROOT:-./data/CelebHQRefForRelease}"
GENDER_METADATA="${GENDER_METADATA:-gender_map.json}"
MODEL_VERSION="${MODEL_VERSION:-sim_stage1}"
SCENE_PACKS_FILE="${SCENE_PACKS_FILE:-scene_packs/scene_packs_large1.json}"
PROMPT_FILE="${PROMPT_FILE:-splits/train_metadata.csv}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
NUM_REPEAT="${NUM_REPEAT:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

INSTANCE_RANK="${SLURM_NODEID:-0}"

mkdir -p logs

echo "============================================"
echo "  InfiniteYou Slurm Data Generation"
echo "============================================"
echo ""
echo "  SLURM_JOB_ID    : ${SLURM_JOB_ID:-}"
echo "  SLURM_NODEID    : $INSTANCE_RANK"
echo "  Hostname         : $(hostname)"
echo "  Model directory  : $MODEL_DIR"
echo "  CelebHQ root     : $CELEB_HQ_ROOT"
echo "  Model version    : $MODEL_VERSION"
echo "  Cluster layout   : $CLUSTER_LAYOUT"
echo ""

# -------------------------------------------------------------------------
# Step 1: Virtual environment
# -------------------------------------------------------------------------
echo ">>> Step 1: Setting up Python environment..."

if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "  Installing requirements..."
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q
echo "  Done."

# -------------------------------------------------------------------------
# Step 2: Download models (node 0 only, unless SKIP_DOWNLOAD=1)
# -------------------------------------------------------------------------
if [ "$SKIP_DOWNLOAD" = "1" ]; then
    echo ""
    echo ">>> Step 2: Skipping model download (SKIP_DOWNLOAD=1)."
elif [ "$INSTANCE_RANK" = "0" ]; then
    echo ""
    echo ">>> Step 2: Downloading models to $MODEL_DIR (node 0)..."
    bash download_models.sh "$MODEL_DIR"
    echo "  Model download complete."
else
    echo ""
    echo ">>> Step 2: Skipping model download (node $INSTANCE_RANK, not node 0)."
    echo "  Assuming models are on shared storage at $MODEL_DIR."
fi

# -------------------------------------------------------------------------
# Step 3: Validate GPUs
# -------------------------------------------------------------------------
echo ""
echo ">>> Step 3: Validating GPUs..."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "  nvidia-smi not found."
    exit 1
fi

NUM_GPUS="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"

IFS=',' read -r -a LAYOUT <<< "$CLUSTER_LAYOUT"
NUM_INSTANCES="${#LAYOUT[@]}"

if [ "$INSTANCE_RANK" -lt 0 ] || [ "$INSTANCE_RANK" -ge "$NUM_INSTANCES" ]; then
    echo "  INSTANCE_RANK=$INSTANCE_RANK is out of range for CLUSTER_LAYOUT=$CLUSTER_LAYOUT"
    exit 1
fi

EXPECTED_GPUS="${LAYOUT[$INSTANCE_RANK]}"

echo "  Detected GPUs: $NUM_GPUS"
echo "  Expected GPUs: $EXPECTED_GPUS (from layout[$INSTANCE_RANK])"

if [ "$NUM_GPUS" -ne "$EXPECTED_GPUS" ]; then
    echo "  ERROR: GPU count mismatch."
    exit 1
fi

# -------------------------------------------------------------------------
# Step 4: Build common arguments
# -------------------------------------------------------------------------
COMMON_ARGS=(
    --model_dir "$MODEL_DIR"
    --celeb_hq_root "$CELEB_HQ_ROOT"
    --celeb_hq_gender_metadata "$GENDER_METADATA"
    --model_version "$MODEL_VERSION"
    --enable_anti_blur_lora2
    --num-samples "$NUM_SAMPLES"
    --num-repeat "$NUM_REPEAT"
    --instance_rank "$INSTANCE_RANK"
    --cluster_layout "$CLUSTER_LAYOUT"
    --resume
    --scene-packs-file "$SCENE_PACKS_FILE"
    --prompt_file "$PROMPT_FILE"
)

if [ -n "$OUTPUT_DIR" ]; then
    COMMON_ARGS+=(--output_dir "$OUTPUT_DIR")
fi

# -------------------------------------------------------------------------
# Step 5: Launch one worker per GPU
# -------------------------------------------------------------------------
echo ""
echo ">>> Step 4: Launching $NUM_GPUS workers on node $INSTANCE_RANK..."

PIDS=()

for ((i=0; i<NUM_GPUS; i++)); do
    LOG_PREFIX="logs/job_${SLURM_JOB_ID:-manual}_node${INSTANCE_RANK}_gpu${i}"
    echo "  Starting worker on GPU $i (log: ${LOG_PREFIX}.{out,err})"

    (
        export CUDA_VISIBLE_DEVICES="$i"
        source "$PWD/$VENV_DIR/bin/activate"

        python3 data_generator.py \
            "${COMMON_ARGS[@]}" \
            --cuda_device "$i" \
            > "${LOG_PREFIX}.out" 2> "${LOG_PREFIX}.err"
    ) &

    PIDS+=("$!")
done

echo "  All $NUM_GPUS workers launched. Waiting for completion..."

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "  Worker PID $pid failed."
        FAIL=$((FAIL + 1))
    fi
done

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "ERROR: $FAIL worker(s) failed on node $INSTANCE_RANK."
    exit 1
fi

echo ""
echo "============================================"
echo "  All workers finished on node $INSTANCE_RANK"
echo "============================================"
