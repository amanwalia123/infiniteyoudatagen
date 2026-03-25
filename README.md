# InfiniteYou Data Generator

Generates identity-preserving synthetic face images at scale using the [InfiniteYou (InfU)](https://github.com/bytedance/InfiniteYou) pipeline built on [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev). Each generated image is quality-gated by an ArcFace similarity score so only images that faithfully preserve the source identity are kept.

## Pipeline Overview

1. Load celebrity reference images from CelebHQ.
2. Build a text prompt per sample — either randomly via `PromptGenerator`, or from a CSV file.
3. Run the InfUFlux pipeline to produce an image matching the prompt while preserving the reference face identity.
4. Compute an ArcFace cosine-similarity score between the reference and generated faces.
5. Keep only images above the configurable similarity threshold and write metadata to disk.

## Repository Structure

```
data_generator.py           # Main generation script
prompt_generator.py         # Random prompt generation with scene packs
utils.py                    # Utility helpers (IP address, server name)
app.py                      # Local Gradio demo
gender_map.json             # Identity → gender mapping for CelebHQ
requirements.txt            # Python dependencies
pipelines/
  pipeline_infu_flux.py     # InfUFlux pipeline wrapper
  pipeline_flux_infusenet.py # InfuseNet FLUX pipeline
  resampler.py              # Resampler module
scene_packs/
  scene_packs_large1.json   # Extended scene descriptions for prompt generation
splits/
  train_metadata.csv        # Training split prompt CSV
  eval_metadata.csv         # Evaluation split prompt CSV
generate_data.sh            # Single-GPU generation script
generate_data2.sh           # Single-GPU generation script (venv setup)
generate_data_multi_gpus.sh # Multi-GPU generation (screen sessions)
generate_data_slurm.sh      # Multi-GPU generation (Slurm)
```

## Model Zoo

| InfiniteYou Version | Model Version | Base Model Trained with | Description |  
| :---: | :---: | :---: | :---: |
| [InfiniteYou-FLUX v1.0](https://huggingface.co/ByteDance/InfiniteYou) | [aes_stage2](https://huggingface.co/ByteDance/InfiniteYou/tree/main/infu_flux_v1.0/aes_stage2) | [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Stage-2 model after SFT. Better text-image alignment and aesthetics. |
| [InfiniteYou-FLUX v1.0](https://huggingface.co/ByteDance/InfiniteYou) | [sim_stage1](https://huggingface.co/ByteDance/InfiniteYou/tree/main/infu_flux_v1.0/sim_stage1) | [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Stage-1 model before SFT. Higher identity similarity. |


## Requirements

### Dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Hardware

| Configuration | Peak VRAM |
|---|---|
| Full `bf16` inference | ~43 GB |
| `--cpu_offload` | ~30 GB |
| `--quantize_8bit` | ~24 GB |
| `--cpu_offload --quantize_8bit` | ~16 GB |

## Quick Start

### Single GPU

```bash
python data_generator.py \
    --celeb_hq_root /path/to/CelebHQRefForRelease \
    --prompt_file splits/train_metadata.csv \
    --output_dir ./output \
    --model_version sim_stage1 \
    --num-repeat 3
```

### Multi-GPU (screen sessions)

```bash
bash generate_data_multi_gpus.sh
```

Launches one `screen` session per GPU. Each GPU processes a non-overlapping slice of the prompt CSV using `--num_gpus` and `--cuda_device` for round-robin partitioning.

### Multi-GPU (Slurm)

```bash
sbatch generate_data_slurm.sh
```

### Gradio Demo

```bash
python app.py
```

## Resuming Interrupted Runs

Generation can be resumed after a crash or preemption with `--resume`:

```bash
python data_generator.py \
    --prompt_file splits/train_metadata.csv \
    --output_dir ./output \
    --num_gpus 4 \
    --cuda_device 0 \
    --resume
```

**How it works:**

- Requires `--prompt_file` — the CSV is the deterministic source of truth (no separate plan file needed).
- Each GPU takes every Nth row (round-robin by `--cuda_device`), so work is partitioned without overlap.
- Each GPU writes to its own output directory, so there are no file conflicts between concurrent processes.
- On `--resume`, the script reads the existing `metadata.json` to discover which `(identity, file_id, repeat)` combinations are already done and skips them.
- Metadata writes use atomic file operations (write-to-temp + `os.replace`) to prevent corruption from mid-write kills.

## CLI Arguments

### Data & Prompt Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--celeb_hq_root` | str | *(see code)* | Path to CelebHQ image directories |
| `--celeb_hq_gender_metadata` | str | `gender_map.json` | Path to the gender metadata JSON |
| `--prompt_file` | str | None | CSV with `prompt`, `identity`, `file_id` columns |
| `--num-samples` | int | 500 | Number of samples to generate (random mode only) |
| `--num-repeat` | int | 3 | Successful repeats per sample |
| `--score_thresh` | float | 0.45 | Minimum ArcFace similarity to keep an image |
| `--force-gender` | str | None | Force gender: `man` or `woman` |
| `--scene-packs-file` | str | None | JSON with additional scene pack descriptions |

### Model & Inference Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--base_model_path` | str | `black-forest-labs/FLUX.1-dev` | Base diffusion model |
| `--model_dir` | str | *(see code)* | Path to InfiniteYou model weights |
| `--model_version` | str | `aes_stage2` | `aes_stage2` (better aesthetics) or `sim_stage1` (higher ID similarity) |
| `--guidance_scale` | float | 3.5 | Classifier-free guidance scale |
| `--num_steps` | int | 30 | Diffusion inference steps |
| `--infusenet_conditioning_scale` | float | 1.0 | InfuseNet conditioning strength |
| `--infusenet_guidance_start` | float | 0.0 | InfuseNet injection start point |
| `--infusenet_guidance_end` | float | 1.0 | InfuseNet injection end point |
| `--img_size` | int int | `1152 1024` | Output image size (width height) |
| `--control_image` | str | None | Optional control image for facial keypoints |

### LoRA Options (optional)

| Argument | Description |
|---|---|
| `--enable_realism_lora` | Enable the [XLabs Realism LoRA](https://civitai.com/models/631986/xlabs-flux-realism-lora?modelVersionId=706528) |
| `--enable_anti_blur_lora` | Enable the [Anti-Blur FLUX LoRA](https://civitai.com/models/675581/anti-blur-flux-lora) |
| `--enable_anti_blur_lora2` | Enable [Shakker-Labs AntiBlur](https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur) LoRA (scale 3.0) |

#### Downloading LoRAs

LoRA weights are expected in `<model_dir>/supports/optional_loras/`. Download them manually and place the `.safetensors` files in that directory.

| LoRA | Source | Expected filename | Flag |
|---|---|---|---|
| XLabs Realism | [CivitAI](https://civitai.com/models/631986/xlabs-flux-realism-lora?modelVersionId=706528) | `flux_realism_lora.safetensors` | `--enable_realism_lora` |
| Anti-Blur FLUX | [CivitAI](https://civitai.com/models/675581/anti-blur-flux-lora) | `flux_anti_blur_lora.safetensors` | `--enable_anti_blur_lora` |
| Shakker-Labs AntiBlur | [HuggingFace](https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur) | `FLUX-dev-lora-AntiBlur.safetensors` | `--enable_anti_blur_lora2` |

The Shakker-Labs LoRA can be downloaded via the CLI:

```bash
pip install huggingface-hub
huggingface-cli download Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur \
    FLUX-dev-lora-AntiBlur.safetensors \
    --local-dir "$MODEL_DIR/supports/optional_loras"
```

The CivitAI LoRAs must be downloaded manually from their pages and placed in the same directory.

### Runtime Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--cuda_device` | int | 0 | GPU device ID |
| `--num_gpus` | int | 1 | Total GPUs for work partitioning |
| `--seed` | int | 0 | Random seed (0 = random) |
| `--output_dir` | str | None | Output directory (auto-derived if omitted) |
| `--resume` | flag | — | Resume from existing `metadata.json` |
| `--quantize_8bit` | flag | — | 8-bit model quantization |
| `--cpu_offload` | flag | — | CPU offloading to reduce VRAM |

## Prompt CSV Format

The `--prompt_file` CSV must have these columns:

```csv
prompt,identity,file_id
"A man, portrait, cinematic",00071,15.png
"A woman in a forest",00233,0.png
```

- **prompt** — text prompt for the diffusion model
- **identity** — subdirectory name under `--celeb_hq_root`  
- **file_id** — filename of the reference image within that subdirectory

## Output

Each run produces:

- **Generated images**: `iden_{identity}_img_{fid}_sample_{N}_repeat_{R}.png`
- **metadata.json**: Append-only JSON array with one entry per accepted image:

```json
{
    "image_name": "iden_00071_img_15_sample_1_repeat_1.png",
    "prompt": "A man, portrait, cinematic",
    "identity": "00071",
    "file_id": "15.png",
    "trial": 1,
    "repeat": 1,
    "similarity_score": 0.62
}
```

## License

Code is released under the [Apache License 2.0](./LICENSE). Model weights follow their respective upstream licenses.
