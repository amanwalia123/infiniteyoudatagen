#!/usr/bin/env bash
#
# run.sh — One script to set up everything and generate data.
#
# Steps:
#   1. Create/activate a Python virtual environment
#   2. Install requirements
#   3. Download all required models (skips if already present)
#   4. Run data generation (single-GPU or multi-GPU)
#
# Usage:
#   bash run.sh [OPTIONS]
#
# Environment variables (all optional, with sensible defaults):
#
#   MODEL_DIR          Where to download / find models   (default: ./models)
#   CELEB_HQ_ROOT      CelebHQ image directory           (default: ./data/CelebHQRefForRelease)
#   GENDER_METADATA    Gender metadata JSON               (default: gender_map.json)
#   MODEL_VERSION      sim_stage1 | aes_stage2            (default: sim_stage1)
#   SCENE_PACKS_FILE   Scene packs JSON                   (default: scene_packs/scene_packs_large1.json)
#   PROMPT_FILE        CSV with prompt/identity/file_id   (default: splits/train_metadata.csv)
#   NUM_SAMPLES        Samples per GPU                    (default: 500)
#   NUM_REPEAT         Repeats per prompt                 (default: 3)
#   OUTPUT_DIR         Output directory                   (default: let data_generator.py decide)
#   INSTANCE_RANK      Cluster instance rank              (default: 0)
#   CLUSTER_LAYOUT     Comma-separated GPU counts         (default: auto-detected)
#   SKIP_DOWNLOAD      Set to 1 to skip model download    (default: 0)
#   MULTI_GPU          Set to 1 to use all GPUs           (default: 0, single GPU)
#   PHYSICAL_GPU       Which GPU for single-GPU mode      (default: 0)
#   HF_TOKEN           HuggingFace token for gated models (default: from env or prompted)
#
# Examples:
#   # Download models to /data/models and generate on one GPU:
#   MODEL_DIR=/data/models CELEB_HQ_ROOT=/data/celebhq bash run.sh
#
#   # Use all GPUs on this machine:
#   MULTI_GPU=1 MODEL_DIR=/data/models CELEB_HQ_ROOT=/data/celebhq bash run.sh
#
#   # Skip download (models already present):
#   SKIP_DOWNLOAD=1 MODEL_DIR=/data/models bash run.sh
#

set -euo pipefail

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
VENV_DIR=".venv"
MODEL_DIR="${MODEL_DIR:-./models}"
CELEB_HQ_ROOT="${CELEB_HQ_ROOT:-./data/CelebHQRefForRelease}"
GENDER_METADATA="${GENDER_METADATA:-gender_map.json}"
MODEL_VERSION="${MODEL_VERSION:-sim_stage1}"
SCENE_PACKS_FILE="${SCENE_PACKS_FILE:-scene_packs/scene_packs_large1.json}"
PROMPT_FILE="${PROMPT_FILE:-splits/train_metadata.csv}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
NUM_REPEAT="${NUM_REPEAT:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
INSTANCE_RANK="${INSTANCE_RANK:-0}"
CLUSTER_LAYOUT="${CLUSTER_LAYOUT:-}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
MULTI_GPU="${MULTI_GPU:-0}"
PHYSICAL_GPU="${PHYSICAL_GPU:-0}"

echo "============================================"
echo "  InfiniteYou Data Generation Pipeline"
echo "============================================"
echo ""
echo "  Model directory : $MODEL_DIR"
echo "  CelebHQ root    : $CELEB_HQ_ROOT"
echo "  Model version   : $MODEL_VERSION"
echo "  Prompt file     : $PROMPT_FILE"
echo "  Multi-GPU       : $MULTI_GPU"
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
# Step 2: Download models (unless SKIP_DOWNLOAD=1)
# -------------------------------------------------------------------------
if [ "$SKIP_DOWNLOAD" = "1" ]; then
    echo ""
    echo ">>> Step 2: Skipping model download (SKIP_DOWNLOAD=1)."
else
    echo ""
    echo ">>> Step 2: Downloading models to $MODEL_DIR..."
    bash download_models.sh "$MODEL_DIR"
    echo "  Model download complete."
fi

# -------------------------------------------------------------------------
# Step 3: Detect GPUs
# -------------------------------------------------------------------------
echo ""
echo ">>> Step 3: Detecting GPUs..."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "  nvidia-smi not found. Assuming CPU-only or MPS."
    NUM_GPUS=1
    MULTI_GPU="0"
else
    NUM_GPUS="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
    echo "  Detected $NUM_GPUS GPU(s)."
fi

# Default cluster layout to detected GPU count if not set
if [ -z "$CLUSTER_LAYOUT" ]; then
    CLUSTER_LAYOUT="$NUM_GPUS"
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
# Step 4: Run generation
# -------------------------------------------------------------------------
echo ""
if [ "$MULTI_GPU" = "1" ] && [ "$NUM_GPUS" -gt 1 ]; then
    echo ">>> Step 4: Launching generation on $NUM_GPUS GPUs..."
    PIDS=()
    for ((i=0; i<NUM_GPUS; i++)); do
        echo "  Starting worker on GPU $i..."
        (
            export CUDA_VISIBLE_DEVICES="$i"
            python3 data_generator.py \
                "${COMMON_ARGS[@]}" \
                --cuda_device "$i"
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
        echo "ERROR: $FAIL worker(s) failed."
        exit 1
    fi
else
    echo ">>> Step 4: Launching generation on GPU $PHYSICAL_GPU..."
    export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
    python3 data_generator.py \
        "${COMMON_ARGS[@]}" \
        --cuda_device "$PHYSICAL_GPU"
fi

echo ""
echo "============================================"
echo "  Generation complete!"
echo "============================================"
