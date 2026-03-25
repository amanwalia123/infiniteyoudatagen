#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
NUM_SAMPLES_PER_GPU=700
SCREEN_NAMES=()

: "${INSTANCE_RANK:?Must set INSTANCE_RANK}"
: "${CLUSTER_LAYOUT:?Must set CLUSTER_LAYOUT, e.g. 4,4,2}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found."
    exit 1
fi

if ! command -v screen >/dev/null 2>&1; then
    echo "screen is not installed."
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

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')

IFS=',' read -r -a LAYOUT <<< "$CLUSTER_LAYOUT"

if [ "$INSTANCE_RANK" -lt 0 ] || [ "$INSTANCE_RANK" -ge "${#LAYOUT[@]}" ]; then
    echo "INSTANCE_RANK=$INSTANCE_RANK is out of range for CLUSTER_LAYOUT=$CLUSTER_LAYOUT"
    exit 1
fi

EXPECTED_GPUS="${LAYOUT[$INSTANCE_RANK]}"

if [ "$NUM_GPUS" -ne "$EXPECTED_GPUS" ]; then
    echo "Local GPU count mismatch."
    echo "Detected via nvidia-smi: $NUM_GPUS"
    echo "Expected from CLUSTER_LAYOUT[$INSTANCE_RANK]: $EXPECTED_GPUS"
    exit 1
fi

echo "Detected $NUM_GPUS GPUs on instance_rank=$INSTANCE_RANK with cluster layout $CLUSTER_LAYOUT."

for ((i=0; i<NUM_GPUS; i++)); do
    SCREEN_NAME="gpu_app_${INSTANCE_RANK}_${i}"
    SCREEN_NAMES+=("$SCREEN_NAME")

    CMD=$(cat <<EOF
cd "$PWD"
source "$PWD/$VENV_DIR/bin/activate"
export CUDA_VISIBLE_DEVICES=$i
python3 data_generator.py \
    --celeb_hq_root /group-volume/Aman-Contents/data/CelebHQRefForRelease \
    --celeb_hq_gender_metadata gender_map.json \
    --model_version sim_stage1 \
    --enable_anti_blur_lora2 \
    --num-samples $NUM_SAMPLES_PER_GPU \
    --num-repeat 3 \
    --cuda_device $i \
    --instance_rank $INSTANCE_RANK \
    --cluster_layout "$CLUSTER_LAYOUT" \
    --resume \
    --scene-packs-file scene_packs/scene_packs_large1.json \
    --prompt_file splits/train_metadata.csv
EOF
)

    echo "Starting local GPU $i in screen session '$SCREEN_NAME'..."
    screen -dmS "$SCREEN_NAME" bash -c "$CMD"
done

echo "Started screen sessions:"
printf ' - %s\n' "${SCREEN_NAMES[@]}"
echo
echo "Use 'screen -ls' to list them."