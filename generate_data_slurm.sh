#!/usr/bin/env bash
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

VENV_DIR=".venv"

# User-provided cluster layout, e.g.:
#   "4,4,2"
: "${CLUSTER_LAYOUT:?Must set CLUSTER_LAYOUT, e.g. 4,4,2}"

PROMPT_FILE="${PROMPT_FILE:-splits/train_metadata.csv}"
CELEB_HQ_ROOT="${CELEB_HQ_ROOT:-/group-volume/Aman-Contents/data/CelebHQRefForRelease}"
GENDER_METADATA="${GENDER_METADATA:-gender_map.json}"
MODEL_VERSION="${MODEL_VERSION:-sim_stage1}"
SCENE_PACKS_FILE="${SCENE_PACKS_FILE:-scene_packs/scene_packs_large1.json}"
NUM_REPEAT="${NUM_REPEAT:-3}"

mkdir -p logs

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found."
    exit 1
fi

INSTANCE_RANK="${SLURM_NODEID:-0}"

IFS=',' read -r -a LAYOUT <<< "$CLUSTER_LAYOUT"
NUM_INSTANCES="${#LAYOUT[@]}"

if [ "$INSTANCE_RANK" -lt 0 ] || [ "$INSTANCE_RANK" -ge "$NUM_INSTANCES" ]; then
    echo "INSTANCE_RANK=$INSTANCE_RANK is out of range for CLUSTER_LAYOUT=$CLUSTER_LAYOUT"
    exit 1
fi

EXPECTED_GPUS="${LAYOUT[$INSTANCE_RANK]}"
NUM_GPUS="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODEID=$INSTANCE_RANK"
echo "Hostname=$(hostname)"
echo "Detected GPUs=$NUM_GPUS"
echo "Expected GPUs for this node from layout=$EXPECTED_GPUS"
echo "Cluster layout=$CLUSTER_LAYOUT"

if [ "$NUM_GPUS" -ne "$EXPECTED_GPUS" ]; then
    echo "Local GPU count mismatch."
    echo "Detected via nvidia-smi: $NUM_GPUS"
    echo "Expected from CLUSTER_LAYOUT[$INSTANCE_RANK]: $EXPECTED_GPUS"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

PIDS=()

for ((i=0; i<NUM_GPUS; i++)); do
    LOG_PREFIX="logs/job_${SLURM_JOB_ID:-manual}_node${INSTANCE_RANK}_gpu${i}"
    echo "Starting worker on node_rank=$INSTANCE_RANK gpu=$i"

    (
        export CUDA_VISIBLE_DEVICES="$i"
        source "$PWD/$VENV_DIR/bin/activate"

        python3 data_generator.py \
            --celeb_hq_root "$CELEB_HQ_ROOT" \
            --celeb_hq_gender_metadata "$GENDER_METADATA" \
            --model_version "$MODEL_VERSION" \
            --enable_anti_blur_lora2 \
            --num-repeat "$NUM_REPEAT" \
            --cuda_device "$i" \
            --instance_rank "$INSTANCE_RANK" \
            --cluster_layout "$CLUSTER_LAYOUT" \
            --resume \
            --scene-packs-file "$SCENE_PACKS_FILE" \
            --prompt_file "$PROMPT_FILE" \
            > "${LOG_PREFIX}.out" 2> "${LOG_PREFIX}.err"
    ) &

    PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "One or more GPU workers failed on node_rank=$INSTANCE_RANK"
    exit 1
fi

echo "All workers finished successfully on node_rank=$INSTANCE_RANK"