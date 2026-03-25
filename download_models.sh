#!/bin/bash
#
# download_models.sh
#
# Downloads all models required by the InfiniteYou data generation pipeline.
# Uses the huggingface_hub Python API (no CLI binary needed).
#
# Usage:
#   bash download_models.sh [MODEL_DIR]
#
# MODEL_DIR defaults to ./models if not provided. The script creates the
# following directory layout:
#
#   MODEL_DIR/
#     infu_flux_v1.0/          # InfiniteYou weights (InfuseNet + image_proj_model)
#       aes_stage2/
#       sim_stage1/
#     supports/
#       insightface/           # antelopev2 face detection models
#         models/
#           antelopev2/
#       optional_loras/        # LoRA adapters (optional)
#

set -euo pipefail

# Use HF_TOKEN from environment or prompt for login below
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGINGFACEHUB_API_TOKEN="${HF_TOKEN}"

MODEL_DIR="${1:-./models}"
echo "Model directory: $MODEL_DIR"
mkdir -p "$MODEL_DIR"

# Ensure huggingface_hub is importable
if ! python3 -c "import huggingface_hub" &>/dev/null; then
    echo "huggingface_hub not found. Installing..."
    pip install huggingface_hub
fi

# Check if already logged in; if not, prompt for login
if ! python3 -c "from huggingface_hub import whoami; whoami()" &>/dev/null; then
    echo ""
    echo "You are not logged in to HuggingFace."
    echo "Some models (e.g. FLUX.1-dev) require authentication."
    echo "You can get a token from: https://huggingface.co/settings/tokens"
    echo ""
    read -p "Enter your HuggingFace token (or press Enter to skip): " -r HF_TOKEN
    if [ -n "$HF_TOKEN" ]; then
        python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
        echo "Logged in successfully."
    else
        echo "Skipping login. Gated models may fail to download."
    fi
else
    echo "HuggingFace: already logged in."
fi

# Helper: download an entire repo
hf_snapshot_download() {
    local repo_id="$1"
    local local_dir="$2"
    local allow_patterns="${3:-}"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${repo_id}',
    local_dir='${local_dir}',
    allow_patterns=${allow_patterns:-None},
)
print('Done.')
"
}

# Helper: download a single file
hf_file_download() {
    local repo_id="$1"
    local filename="$2"
    local local_dir="$3"
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='${repo_id}',
    filename='${filename}',
    local_dir='${local_dir}',
)
print('Done.')
"
}

# -------------------------------------------------------------------------
# 1. FLUX.1-dev base model (downloaded automatically by diffusers on first
#    run, but you can pre-download it to avoid delays)
# -------------------------------------------------------------------------
echo ""
echo "=== 1/4  FLUX.1-dev base model ==="
echo "The base model (black-forest-labs/FLUX.1-dev) is gated on HuggingFace."
echo "Make sure you have:"
echo "  1. Accepted the license at: https://huggingface.co/black-forest-labs/FLUX.1-dev"
echo "  2. Enabled 'Access to public gated repos' in your token settings:"
echo "     https://huggingface.co/settings/tokens"
echo ""
read -p "Download FLUX.1-dev base model? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if hf_snapshot_download "black-forest-labs/FLUX.1-dev" "$MODEL_DIR/FLUX.1-dev"; then
        echo "FLUX.1-dev downloaded to $MODEL_DIR/FLUX.1-dev"
    else
        echo "ERROR: Failed to download FLUX.1-dev. Check that your token has"
        echo "       'Access to public gated repos' enabled and you accepted the license."
        echo "       Continuing with remaining downloads..."
    fi
else
    echo "Skipped. Diffusers will download it automatically on first run."
fi

# -------------------------------------------------------------------------
# 2. InfiniteYou model weights (InfuseNet + image_proj_model)
# -------------------------------------------------------------------------
echo ""
echo "=== 2/4  InfiniteYou weights ==="
hf_snapshot_download "ByteDance/InfiniteYou" "$MODEL_DIR"
echo "InfiniteYou weights downloaded to $MODEL_DIR"

# -------------------------------------------------------------------------
# 3. InsightFace antelopev2 (face detection used by the pipeline)
# -------------------------------------------------------------------------
echo ""
echo "=== 3/4  InsightFace antelopev2 ==="
INSIGHTFACE_DIR="$MODEL_DIR/supports/insightface/models/antelopev2"
mkdir -p "$INSIGHTFACE_DIR"

if [ -f "$INSIGHTFACE_DIR/1k3d68.onnx" ]; then
    echo "antelopev2 already present, skipping."
else
    echo "Downloading antelopev2 from HuggingFace..."
    hf_snapshot_download "ByteDance/InfiniteYou" "$MODEL_DIR" "['supports/insightface/**']"
    echo "antelopev2 downloaded to $INSIGHTFACE_DIR"
fi

# -------------------------------------------------------------------------
# 4. Optional LoRA adapters
# -------------------------------------------------------------------------
echo ""
echo "=== 4/4  Optional LoRA adapters ==="
LORA_DIR="$MODEL_DIR/supports/optional_loras"
mkdir -p "$LORA_DIR"

# --- Shakker-Labs AntiBlur LoRA (--enable_anti_blur_lora2) ---
ANTIBLUR2_FILE="$LORA_DIR/FLUX-dev-lora-AntiBlur.safetensors"
if [ -f "$ANTIBLUR2_FILE" ]; then
    echo "Shakker-Labs AntiBlur LoRA already present, skipping."
else
    echo "Downloading Shakker-Labs AntiBlur LoRA..."
    hf_file_download "Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur" "FLUX-dev-lora-AntiBlur.safetensors" "$LORA_DIR"
    echo "Saved to $ANTIBLUR2_FILE"
fi

# --- CivitAI LoRAs (manual download required) ---
echo ""
REALISM_FILE="$LORA_DIR/flux_realism_lora.safetensors"
ANTIBLUR_FILE="$LORA_DIR/flux_anti_blur_lora.safetensors"

if [ ! -f "$REALISM_FILE" ]; then
    echo "[Manual] XLabs Realism LoRA (--enable_realism_lora):"
    echo "  Download from: https://civitai.com/models/631986/xlabs-flux-realism-lora?modelVersionId=706528"
    echo "  Save as:       $REALISM_FILE"
else
    echo "XLabs Realism LoRA already present."
fi

if [ ! -f "$ANTIBLUR_FILE" ]; then
    echo "[Manual] Anti-Blur FLUX LoRA (--enable_anti_blur_lora):"
    echo "  Download from: https://civitai.com/models/675581/anti-blur-flux-lora"
    echo "  Save as:       $ANTIBLUR_FILE"
else
    echo "Anti-Blur FLUX LoRA already present."
fi

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Download complete. Model directory layout:"
echo "========================================"
echo ""
echo "  $MODEL_DIR/"
echo "    FLUX.1-dev/                  # Base diffusion model (if downloaded)"
echo "    infu_flux_v1.0/"
echo "      aes_stage2/               # InfuseNet + image_proj_model (aesthetics)"
echo "      sim_stage1/               # InfuseNet + image_proj_model (similarity)"
echo "    supports/"
echo "      insightface/models/       # antelopev2 face detection"
echo "      optional_loras/           # LoRA adapters"
echo ""
echo "Usage:"
echo "  python data_generator.py --model_dir $MODEL_DIR --base_model_path $MODEL_DIR/FLUX.1-dev ..."
echo ""
