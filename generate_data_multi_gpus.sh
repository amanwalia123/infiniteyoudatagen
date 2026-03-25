#!/bin/bash

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate
echo "Installing requirements..."
pip install -r requirements.txt

NUM_SAMPLES_PER_GPU=700
SCREEN_NAMES=()

# Determine the number of GPUs by counting the lines from the output of nvidia-smi --list-gpus
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# Check if the number of GPUs was retrieved successfully and is greater than 0
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Could not find any GPUs or nvidia-smi is not available."
    exit 1
fi

echo "Detected $NUM_GPUS GPUs."

# Loop through each GPU index
for ((i=0; i<NUM_GPUS; i++)); do
    SCREEN_NAME="gpu_app_$i"
    SCREEN_NAMES+=("$SCREEN_NAME")
    echo "Starting process for GPU $i in screen session '$SCREEN_NAME'..."

    CMD="source \$PWD/\$VENV_DIR/bin/activate && CUDA_VISIBLE_DEVICES=\$i python3 data_generator.py --celeb_hq_root /group-volume/Aman-Contents/data/CelebHQRefForRelease \
                                                                                                                      --celeb_hq_gender_metadata  gender_map.json \
                                                                                                                      --model_version sim_stage1 \
                                                                                                                      --enable_anti_blur_lora2 \
                                                                                                                      --num-samples $NUM_SAMPLES_PER_GPU \
                                                                                                                      --num-repeat 3 \
                                                                                                                      --cuda_device $i \
                                                                                                                      --scene-packs-file scene_packs/scene_packs_large1.json \
                                                                                                                      --prompt_file splits/train_metadata.csv"

    # Launch the screen session in the background
    screen -dmS "$SCREEN_NAME" bash -c "$CMD"
done
