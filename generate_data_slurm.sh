#!/bin/bash
# generate_data_slurm.sh
#
# SLURM batch script for distributed data generation across multiple nodes
# and GPUs using torch.distributed (NCCL backend).
#
# Submit with:
#   sbatch generate_data_slurm.sh
#
# Or override defaults at submission time, e.g.:
#   sbatch --nodes=4 --ntasks-per-node=8 generate_data_slurm.sh

# ---------------------------------------------------------------------------
# SLURM resource directives (edit these to match your cluster)
# ---------------------------------------------------------------------------
#SBATCH --job-name=infiniteyou_datagen
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.log
#SBATCH --error=logs/slurm/%x_%j.log
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
VENV_PATH="${VENV_PATH:-/home/user/.virtualenvs/infiniteyou}"
source "${VENV_PATH}/bin/activate"

# ---------------------------------------------------------------------------
# Distributed training configuration derived from SLURM variables
# ---------------------------------------------------------------------------
# Number of tasks (processes) per node
NTASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-${SLURM_GPUS_PER_TASK:-1}}"
# Total number of processes across all nodes
WORLD_SIZE=$(( SLURM_NNODES * NTASKS_PER_NODE ))
# Use the first node as the master for rendezvous
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT="${MASTER_PORT:-29500}"

echo "========================================="
echo "SLURM Job Info"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Nodes        : ${SLURM_NNODES}"
echo "  Tasks/node   : ${NTASKS_PER_NODE}"
echo "  World size   : ${WORLD_SIZE}"
echo "  Master addr  : ${MASTER_ADDR}:${MASTER_PORT}"
echo "========================================="

# Create the log directory if it does not exist yet
mkdir -p logs/slurm

# ---------------------------------------------------------------------------
# Data generation parameters (override via environment variables)
# ---------------------------------------------------------------------------
CELEB_HQ_ROOT="${CELEB_HQ_ROOT:-/group-volume/Aman-Contents/data/CelebHQRefForRelease}"
CELEB_HQ_METADATA="${CELEB_HQ_METADATA:-${CELEB_HQ_ROOT}/gender_map.json}"
MODEL_VERSION="${MODEL_VERSION:-sim_stage1}"
SCENE_PACKS_FILE="${SCENE_PACKS_FILE:-scene_packs/scene_packs_large1.json}"
NUM_SAMPLES_PER_GPU="${NUM_SAMPLES_PER_GPU:-700}"
NUM_REPEAT="${NUM_REPEAT:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

OUTPUT_ARG=""
if [[ -n "${OUTPUT_DIR}" ]]; then
    OUTPUT_ARG="--output_dir ${OUTPUT_DIR}"
fi

# ---------------------------------------------------------------------------
# Launch one process per GPU using srun
# Each process receives its own SLURM_PROCID which maps to LOCAL_RANK/RANK.
# ---------------------------------------------------------------------------
srun --ntasks="${WORLD_SIZE}" \
     --ntasks-per-node="${NTASKS_PER_NODE}" \
     --gpus-per-task=1 \
     bash -c "
set -euo pipefail

# srun sets SLURM_PROCID (global rank) and SLURM_LOCALID (local rank on node)
RANK=\${SLURM_PROCID}
LOCAL_RANK=\${SLURM_LOCALID}

echo \"[Rank \${RANK}] Node: \$(hostname), Local rank: \${LOCAL_RANK}, CUDA: \${LOCAL_RANK}\"

CUDA_VISIBLE_DEVICES=\${LOCAL_RANK} python3 data_generator.py \
    --celeb_hq_root ${CELEB_HQ_ROOT} \
    --celeb_hq_gender_metadata ${CELEB_HQ_METADATA} \
    --model_version ${MODEL_VERSION} \
    --enable_anti_blur_lora2 \
    --num-samples ${NUM_SAMPLES_PER_GPU} \
    --num-repeat ${NUM_REPEAT} \
    --scene-packs-file ${SCENE_PACKS_FILE} \
    --distributed \
    --cuda_device \${LOCAL_RANK} \
    --dist_rank \${RANK} \
    --dist_world_size ${WORLD_SIZE} \
    --dist_master_addr ${MASTER_ADDR} \
    --dist_master_port ${MASTER_PORT} \
    ${OUTPUT_ARG}

echo \"[Rank \${RANK}] Finished.\"
"

echo "All ranks completed successfully."
