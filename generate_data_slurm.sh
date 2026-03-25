#!/bin/bash
#SBATCH --job-name=gen_multi_gpu
#SBATCH --output=logs/gen_multi_gpu_%j.out
#SBATCH --error=logs/gen_multi_gpu_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8   # Adjust depending on available GPUs per node
#SBATCH --cpus-per-task=32  # Use enough CPU cores
#SBATCH --mem=256G          # Memory for the entire job

mkdir -p logs

NUM_SAMPLES_PER_GPU=700

# Determine the number of GPUs from Slurm environment or fallback to nvidia-smi
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
else
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi

# Check if the number of GPUs was retrieved successfully and is greater than 0
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Could not find any GPUs or nvidia-smi is not available."
    exit 1
fi

echo "Detected $NUM_GPUS GPUs."

source /home/user/.virtualenvs/infiniteyou/bin/activate 

PROMPT_FILES=("splits/train_metadata.csv" "splits/eval_metadata.csv")

for PROMPT_FILE in "${PROMPT_FILES[@]}"; do
    echo "Starting generation for $PROMPT_FILE..."

    # Loop through each GPU index
    for ((i=0; i<NUM_GPUS; i++)); do
        echo "Starting process for GPU $i with $PROMPT_FILE in the background..."

        CUDA_VISIBLE_DEVICES=$i python3 data_generator.py \
            --celeb_hq_root /group-volume/Aman-Contents/data/CelebHQRefForRelease \
            --celeb_hq_gender_metadata /group-volume/Aman-Contents/data/CelebHQRefForRelease/gender_map.json \
            --model_version sim_stage1 \
            --enable_anti_blur_lora2 \
            --num-samples $NUM_SAMPLES_PER_GPU \
            --num-repeat 3 \
            --cuda_device $i \
            --scene-packs-file scene_packs/scene_packs_large1.json \
            --prompt_file $PROMPT_FILE &
    done

    echo "Waiting for all processes for $PROMPT_FILE to finish..."
    wait
done

echo "All done!"
