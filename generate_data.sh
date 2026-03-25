#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

# Physical GPU to expose to this process
PHYSICAL_GPU="${PHYSICAL_GPU:-0}"

# Logical local GPU index used in cluster sharding math
CUDA_DEVICE="${CUDA_DEVICE:-0}"

PROMPT_FILE="${PROMPT_FILE:-splits/train_metadata.csv}"
CELEB_HQ_ROOT="${CELEB_HQ_ROOT:-/group-volume/Aman-Contents/data/CelebHQRefForRelease}"
GENDER_METADATA="${GENDER_METADATA:-gender_map.json}"
MODEL_VERSION="${MODEL_VERSION:-sim_stage1}"
SCENE_PACKS_FILE="${SCENE_PACKS_FILE:-scene_packs/scene_packs_large1.json}"
NUM_REPEAT="${NUM_REPEAT:-3}"
INSTANCE_RANK="${INSTANCE_RANK:-0}"
CLUSTER_LAYOUT="${CLUSTER_LAYOUT:-1}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found."
    exit 1
fi

NUM_GPUS="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"

if [ "$PHYSICAL_GPU" -lt 0 ] || [ "$PHYSICAL_GPU" -ge "$NUM_GPUS" ]; then
    echo "Invalid PHYSICAL_GPU=$PHYSICAL_GPU. Detected $NUM_GPUS GPUs."
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing requirements..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"

python3 data_generator.py \
    --celeb_hq_root "$CELEB_HQ_ROOT" \
    --celeb_hq_gender_metadata "$GENDER_METADATA" \
    --model_version "$MODEL_VERSION" \
    --enable_anti_blur_lora2 \
    --num-repeat "$NUM_REPEAT" \
    --cuda_device "$CUDA_DEVICE" \
    --instance_rank "$INSTANCE_RANK" \
    --cluster_layout "$CLUSTER_LAYOUT" \
    --resume \
    --scene-packs-file "$SCENE_PACKS_FILE" \
    --prompt_file "$PROMPT_FILE"