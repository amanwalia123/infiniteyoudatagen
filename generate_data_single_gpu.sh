#!/usr/bin/env bash
set -euo pipefail

NUM_SAMPLES=700

echo "Installing requirements..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Check if nvidia-smi is available
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found."
    exit 1
fi

# Check if at least one GPU is available
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "No GPUs detected."
    exit 1
fi

echo "Detected $NUM_GPUS GPUs. Using GPU 0 for generation."

# Run the data generator on GPU 0
export CUDA_VISIBLE_DEVICES=0
python3 data_generator.py \
    --celeb_hq_root /netapp/output/aman.walia/data/CelebHQRefForRelease \
    --model_dir /netapp/output/aman.walia/models/InfiniteYou \
    --celeb_hq_gender_metadata gender_map.json \
    --model_version sim_stage1 \
    --enable_anti_blur_lora2 \
    --num-samples $NUM_SAMPLES \
    --num-repeat 3 \
    --cuda_device 0 \
    --resume \
    --scene-packs-file scene_packs/scene_packs_large1.json \
    --model_dir /netapp/output/aman.walia/models/InfiniteYou \
    --prompt_file splits/train_metadata.csv

echo "Data generation completed on GPU 0."