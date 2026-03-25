#!/bin/bash
# generate_data_multi_gpus.sh
#
# Submits one SLURM job per GPU (or per node when running multi-node) so that
# every GPU independently generates data via data_generator.py.
#
# Usage:
#   bash generate_data_multi_gpus.sh [--nodes N] [--gpus-per-node G]
#                                    [--num-samples S] [--num-repeat R]
#                                    [--output-dir DIR] [--partition PARTITION]
#
# When SLURM is not available the script falls back to launching a plain
# background process per detected GPU (mirrors the original screen behaviour).

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (override via command-line flags or environment variables)
# ---------------------------------------------------------------------------
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-0}"          # 0 = auto-detect
NUM_SAMPLES_PER_GPU="${NUM_SAMPLES_PER_GPU:-700}"
NUM_REPEAT="${NUM_REPEAT:-3}"
PARTITION="${PARTITION:-gpu}"
VENV_PATH="${VENV_PATH:-/home/user/.virtualenvs/infiniteyou}"
CELEB_HQ_ROOT="${CELEB_HQ_ROOT:-/group-volume/Aman-Contents/data/CelebHQRefForRelease}"
CELEB_HQ_METADATA="${CELEB_HQ_METADATA:-${CELEB_HQ_ROOT}/gender_map.json}"
MODEL_VERSION="${MODEL_VERSION:-sim_stage1}"
SCENE_PACKS_FILE="${SCENE_PACKS_FILE:-scene_packs/scene_packs_large1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
LOG_DIR="${LOG_DIR:-logs/slurm}"

# ---------------------------------------------------------------------------
# Parse optional command-line arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)        NODES="$2";           shift 2 ;;
        --gpus-per-node) GPUS_PER_NODE="$2";  shift 2 ;;
        --num-samples)  NUM_SAMPLES_PER_GPU="$2"; shift 2 ;;
        --num-repeat)   NUM_REPEAT="$2";      shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";      shift 2 ;;
        --partition)    PARTITION="$2";       shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Detect number of GPUs on this node if not specified
# ---------------------------------------------------------------------------
if [[ "$GPUS_PER_NODE" -eq 0 ]]; then
    if command -v nvidia-smi &>/dev/null; then
        GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
    else
        echo "ERROR: nvidia-smi not found and --gpus-per-node was not set." >&2
        exit 1
    fi
fi

if [[ "$GPUS_PER_NODE" -eq 0 ]]; then
    echo "ERROR: No GPUs detected on this node." >&2
    exit 1
fi

echo "Configuration:"
echo "  Nodes            : ${NODES}"
echo "  GPUs per node    : ${GPUS_PER_NODE}"
echo "  Samples per GPU  : ${NUM_SAMPLES_PER_GPU}"
echo "  Repeats          : ${NUM_REPEAT}"
echo "  SLURM partition  : ${PARTITION}"
echo "  Log dir          : ${LOG_DIR}"

mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Build the common data_generator.py arguments
# ---------------------------------------------------------------------------
OUTPUT_ARG=""
if [[ -n "${OUTPUT_DIR}" ]]; then
    OUTPUT_ARG="--output_dir ${OUTPUT_DIR}"
fi

COMMON_ARGS="--celeb_hq_root ${CELEB_HQ_ROOT} \
             --celeb_hq_gender_metadata ${CELEB_HQ_METADATA} \
             --model_version ${MODEL_VERSION} \
             --enable_anti_blur_lora2 \
             --num-samples ${NUM_SAMPLES_PER_GPU} \
             --num-repeat ${NUM_REPEAT} \
             --scene-packs-file ${SCENE_PACKS_FILE} \
             --distributed \
             ${OUTPUT_ARG}"

# ---------------------------------------------------------------------------
# Submit via SLURM when sbatch is available; otherwise fall back to
# plain background processes (one per GPU, analogous to the original
# screen-session approach).
# ---------------------------------------------------------------------------
if command -v sbatch &>/dev/null; then
    echo "SLURM detected – submitting ${NODES} job(s) (${GPUS_PER_NODE} GPU(s) each)..."

    for ((node_idx=0; node_idx<NODES; node_idx++)); do
        JOB_NAME="datagen_node${node_idx}"
        LOG_FILE="${LOG_DIR}/${JOB_NAME}_%j.log"

        JOB_ID=$(sbatch \
            --job-name="${JOB_NAME}" \
            --partition="${PARTITION}" \
            --nodes=1 \
            --ntasks="${GPUS_PER_NODE}" \
            --gpus-per-task=1 \
            --cpus-per-task=4 \
            --mem-per-gpu=32G \
            --time=24:00:00 \
            --output="${LOG_FILE}" \
            --error="${LOG_FILE}" \
            --parsable \
            --wrap="
source ${VENV_PATH}/bin/activate
export MASTER_ADDR=\$(hostname -I | awk '{print \$1}')
export MASTER_PORT=\$((29500 + ${node_idx}))
export WORLD_SIZE=\$((${NODES} * ${GPUS_PER_NODE}))
export NODE_RANK=${node_idx}
echo \"[Node ${node_idx}] MASTER_ADDR=\${MASTER_ADDR} WORLD_SIZE=\${WORLD_SIZE}\"

for ((gpu=0; gpu<${GPUS_PER_NODE}; gpu++)); do
    GLOBAL_RANK=\$((${node_idx} * ${GPUS_PER_NODE} + gpu))
    echo \"[Node ${node_idx}] Launching rank \${GLOBAL_RANK} on GPU \${gpu}\"
    CUDA_VISIBLE_DEVICES=\${gpu} python3 data_generator.py \
        ${COMMON_ARGS} \
        --cuda_device \${gpu} \
        --dist_rank \${GLOBAL_RANK} \
        --dist_world_size \${WORLD_SIZE} \
        --dist_master_addr \${MASTER_ADDR} \
        --dist_master_port \${MASTER_PORT} &
done
wait
echo \"[Node ${node_idx}] All GPU processes finished.\"
")
        echo "  Submitted job ${JOB_ID} for node index ${node_idx} (log: ${LOG_FILE/\%j/${JOB_ID}})"
    done

    echo "All jobs submitted. Monitor with: squeue -u \$(whoami)"

else
    # -----------------------------------------------------------------
    # Fallback: launch one background process per GPU on this machine
    # -----------------------------------------------------------------
    echo "SLURM not found – launching ${GPUS_PER_NODE} background process(es) locally..."

    TOTAL_GPUS="${GPUS_PER_NODE}"
    MASTER_ADDR="127.0.0.1"
    MASTER_PORT=29500
    PIDS=()

    for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
        LOG_FILE="${LOG_DIR}/gpu_${gpu}.log"
        echo "  Starting process for GPU ${gpu} (log: ${LOG_FILE})..."

        CUDA_VISIBLE_DEVICES=${gpu} python3 data_generator.py \
            ${COMMON_ARGS} \
            --cuda_device ${gpu} \
            --dist_rank ${gpu} \
            --dist_world_size ${TOTAL_GPUS} \
            --dist_master_addr ${MASTER_ADDR} \
            --dist_master_port ${MASTER_PORT} \
            > "${LOG_FILE}" 2>&1 &
        PIDS+=($!)
    done

    echo "All ${TOTAL_GPUS} process(es) launched. PIDs: ${PIDS[*]}"
    echo "Waiting for all processes to complete..."

    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait "${pid}"; then
            echo "ERROR: Process ${pid} failed." >&2
            FAILED=1
        fi
    done

    if [[ "${FAILED}" -eq 0 ]]; then
        echo "All processes completed successfully."
    else
        echo "One or more processes failed. Check logs in ${LOG_DIR}." >&2
        exit 1
    fi
fi
