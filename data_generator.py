"""
data_generator.py

Generates identity-preserving synthetic face images using the InfiniteYou (InfU)
pipeline built on FLUX. The workflow is:

  1. Load celebrity reference images from CelebHQ.
  2. Build a text prompt per sample (randomly via PromptGenerator, or from a CSV).
  3. Run the InfUFluxPipeline to produce an image that matches the prompt while
     preserving the identity of the reference face.
  4. Compute an ArcFace cosine-similarity score between the reference and
     generated faces; only keep images above a configurable threshold.
  5. Write accepted images and their metadata (prompt, identity, score, etc.)
     to disk.

Usage:
    python data_generator.py --output_dir ./output --num-samples 100
    See build_argparser() for the full list of CLI arguments.
"""

import argparse
import csv
import json
import os
import ssl
import random
import tempfile
from glob import glob
import torch
from PIL import Image
import cv2
import numpy as np
from insightface.app import FaceAnalysis

from utils import get_ip_address, get_server_name
from pipelines.pipeline_infu_flux import InfUFluxPipeline
from prompt_generator import GenConfig, PromptGenerator

# ---------------------------------------------------------------------------
# Global SSL workaround – required for downloading HuggingFace model weights
# in environments with custom/broken certificate chains.
# ---------------------------------------------------------------------------
os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context

# ---------------------------------------------------------------------------
# ArcFace face-analysis model (buffalo_l) used to compute identity similarity
# between the source reference image and every generated image.
# Loaded once at module level so it is reused across all samples.
# ---------------------------------------------------------------------------
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(320, 320))

# Maximum number of generation attempts per prompt/repeat before giving up
# (guards against infinite loops when the model keeps producing low-score images).
MAX_RETRIES = 10


def atomic_json_append(json_path, entry):
    """Append *entry* to the JSON array at *json_path* using an atomic write.

    Reads the existing array (or starts a new one), appends the entry, writes
    to a temporary file in the same directory, then atomically replaces the
    target via ``os.replace``.  This prevents corruption if the process is
    killed mid-write.
    """
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    dir_name = os.path.dirname(json_path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=4)
        os.replace(tmp_path, json_path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def load_completed_work(output_dir):
    """Read ``metadata.json`` and return the set of already-completed work items.

    Each completed item is represented as a tuple
    ``(identity, file_id_stem, repeat)`` so we can quickly check whether a
    particular sample/repeat combination has already been accepted.

    Returns:
        set[tuple]: Completed ``(identity, fid, repeat)`` triples.
    """
    json_output_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(json_output_path):
        return set()
    with open(json_output_path, 'r') as f:
        existing_data = json.load(f)
    completed = set()
    for entry in existing_data:
        completed.add((entry["identity"], entry["file_id"].split(".")[0], entry["repeat"]))
    return completed


def build_argparser():
    """Define and parse all CLI arguments.

    Arguments are grouped into:
      - CelebHQ dataset & prompt generation options
      - InfiniteYou model & inference options
      - Optional LoRA adapters
      - Memory-reduction flags (quantisation, CPU offload)
      - Output configuration

    Returns:
        argparse.Namespace: Validated argument namespace.
    """
    parser = argparse.ArgumentParser(description="Generate realistic data from celebrity faces and text prompts for realistic faces in the scenes")

    # --- CelebHQ dataset & prompt generation arguments ---
    parser.add_argument("--celeb_hq_root", type=str, 
                        default="/netapp/output/aman.walia/data/CelebHQRefForRelease", 
                        help="Path to the directory containing CelebHQ image dirs")
    parser.add_argument("--celeb_hq_gender_metadata", type=str, 
                        default="gender_map.json", 
                        help="Path to the gender meta file")
    parser.add_argument("--num-samples",
                        type=int,
                        default=500,
                        required=False,
                        help="number of dataset samples to generate")
    parser.add_argument("--num-repeat",
                        type=int,
                        default=3,
                        required=False,
                        help="num of times to repeat with the same prompt")
    
    parser.add_argument("--force-gender", type=str, choices=["man", "woman"])
    parser.add_argument("--score_thresh", type=float, default=0.45)
    parser.add_argument("--flux-positive-focus", 
                        "--flux", 
                        action="store_true", 
                        help="Use positive-only deep-focus phrasing for Flux Dev.")
    parser.add_argument("--scene-packs-file", 
                        type=str, 
                        default=None, 
                        help="(Legacy) combined JSON with multiple packs.")
    parser.add_argument("--prompt_file", 
                        type=str, 
                        default=None, 
                        help="Path to CSV containing prompt, identity, and file_id")

    # --- InfiniteYou model & inference arguments ---
    parser.add_argument('--base_model_path', default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--model_dir', default='/group-volume/Aman-Contents/InfiniteYou')
    parser.add_argument('--control_image', default=None, help="control image [optional]")
    parser.add_argument('--infu_flux_version', default='v1.0', help="""InfiniteYou-FLUX version: currently only v1.0""")
    parser.add_argument('--model_version', default='aes_stage2', help="""model version: aes_stage2 | sim_stage1""")
    parser.add_argument('--guidance_scale', default=3.5, type=float)
    parser.add_argument('--num_steps', default=30, type=int)
    parser.add_argument('--infusenet_conditioning_scale', default=1.0, type=float)
    parser.add_argument('--infusenet_guidance_start', default=0.0, type=float)
    parser.add_argument('--infusenet_guidance_end', default=1.0, type=float)
    parser.add_argument('--img_size', default=(1152, 1024), type=int, nargs=2, help="""Image size for the generated images (width, height)""")

    # --- Optional LoRA adapters (not used in the original paper) ---
    parser.add_argument('--enable_realism_lora', action='store_true')
    parser.add_argument('--enable_anti_blur_lora', action='store_true')
    parser.add_argument('--enable_anti_blur_lora2', action='store_true')

    # --- Memory-reduction options ---
    parser.add_argument('--quantize_8bit', action='store_true')
    parser.add_argument('--cpu_offload', action='store_true')

    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int, help="""seed (0 for random)""")

    parser.add_argument('--output_dir', type=str, default=None, help="Path to the output directory for generated images")
    parser.add_argument('--resume', action='store_true',
                        help="Resume a previously interrupted run. Requires --prompt_file. "
                             "Skips samples/repeats already present in metadata.json.")
    parser.add_argument('--num_gpus', type=int, default=1,
                        help="Total number of GPUs running in parallel. Each GPU (identified "
                             "by --cuda_device) processes every num_gpus-th row from the "
                             "prompt file so work is evenly partitioned with no overlap.")

    args = parser.parse_args()
    assert args.infu_flux_version == 'v1.0', 'Currently only supports InfiniteYou-FLUX v1.0'
    assert args.model_version in ['aes_stage2', 'sim_stage1'], 'Currently only supports model versions: aes_stage2 | sim_stage1'
    if args.resume and args.prompt_file is None:
        parser.error("--resume requires --prompt_file (resume is only supported with a deterministic prompt CSV)")
    if args.num_gpus > 1 and args.prompt_file is None:
        parser.error("--num_gpus > 1 requires --prompt_file for deterministic work partitioning")
    
    return args


def main():
    """Entry point: parse args, load models, generate images, and filter by identity score."""
    args = build_argparser()

    # Detect which server/cluster we are running on so we can pick a default
    # output directory if none was explicitly provided.
    server_name = get_server_name()
    ip_addr = get_ip_address()

    # Collect all CelebHQ reference images (used when no prompt_file is given).
    celeb_hq_image_paths = glob(f"{args.celeb_hq_root}/**/*.png", recursive=True)

    # Map identity folder names → gender strings ("man" / "woman") so the
    # prompt generator can use gender-appropriate language.
    with open(args.celeb_hq_gender_metadata) as f:
        gender_data = json.load(f)

    # Build the prompt-generation configuration with sensible defaults.
    cfg = GenConfig(
        seed=args.seed,
        num=1,
        force_gender="person",
        from_attr=None,
        scene_packs_file=args.scene_packs_file,
        scene_dir=None,
        scene_pack_filters=None,
        diverse=True,
        out=None,
        min_size=1024,
        flux_positive=True,
        no_negative=True,
    )

    # Pin to GPU 0 for the current process (multi-GPU runs launch separate
    # processes with different --cuda_device values).
    torch.cuda.set_device(0)

    # Derive a server-specific output directory when none is provided.
    if args.output_dir is None:
        if server_name == "lambda":
            root = "/netapp/output/aman.walia/data/infiniteyou_face_dataset"
        elif server_name == "MLP":
            root = "/group-volume/TorAIC-Image-Quality/Aman-Contents/data/infiniteyou_face_dataset"
        elif server_name == "SPACE":
            root = "/group-volume/Aman-Contents/data/infiniteyou_face_dataset"
            
        device = args.cuda_device
        args.output_dir = os.path.join(root, ip_addr, f"device_{device}")

    # -----------------------------------------------------------------------
    # Load the InfiniteYou-FLUX pipeline (base diffusion model + InfuseNet).
    # -----------------------------------------------------------------------
    infu_model_path = os.path.join(args.model_dir, f'infu_flux_{args.infu_flux_version}', args.model_version)
    insightface_root_path = os.path.join(args.model_dir, 'supports', 'insightface')
    pipe = InfUFluxPipeline(
        base_model_path=args.base_model_path,
        infu_model_path=infu_model_path,
        insightface_root_path=insightface_root_path,
        infu_flux_version=args.infu_flux_version,
        model_version=args.model_version,
        quantize_8bit=args.quantize_8bit,
        cpu_offload=args.cpu_offload,
    )

    # Optionally attach LoRA adapters (realism / anti-blur) to the pipeline.
    # Falls back to a hardcoded path if the model_dir copy doesn't exist.
    lora_dir = os.path.join(args.model_dir, 'supports', 'optional_loras')
    if not os.path.exists(lora_dir): lora_dir = '/group-volume/Aman-Contents/InfiniteYou/supports/optional_loras'
    loras = []
    
    if args.enable_realism_lora:
        loras.append([os.path.join(lora_dir, 'flux_realism_lora.safetensors'), 'realism', 1.0])
    
    if args.enable_anti_blur_lora:
        loras.append([os.path.join(lora_dir, 'flux_anti_blur_lora.safetensors'), 'anti_blur', 1.0])
    
    if args.enable_anti_blur_lora2:
        loras.append([os.path.join(lora_dir, 'FLUX-dev-lora-AntiBlur.safetensors'), 'anti_blur', 3.0])
    
    pipe.load_loras(loras)

    # If no explicit seed was given, generate a random one.
    if args.seed == 0:
        args.seed = torch.seed() & 0xFFFFFFFF

    generator = PromptGenerator(cfg)

    # -----------------------------------------------------------------------
    # Build the work list for this GPU.
    #
    # When --prompt_file is given the CSV is the single source of truth.
    # With --num_gpus > 1 each GPU takes a non-overlapping slice of the rows
    # (round-robin by cuda_device).  On --resume we read the per-GPU
    # metadata.json to discover which samples/repeats are already done.
    #
    # Without --prompt_file the original random-sampling path is used.
    # Resume and multi-GPU partitioning are NOT supported in that mode
    # (enforced by argument validation above).
    # -----------------------------------------------------------------------
    if args.prompt_file is not None:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)

        # Partition rows across GPUs: GPU k takes rows k, k+N, k+2N, …
        my_rows = all_rows[args.cuda_device::args.num_gpus]
        print(f"GPU {args.cuda_device}: assigned {len(my_rows)}/{len(all_rows)} rows "
              f"(num_gpus={args.num_gpus})")

        plan = []
        for row in my_rows:
            identity = row['identity']
            file_id = row['file_id']
            plan.append({
                "identity": identity,
                "file_id": file_id,
                "fpath": os.path.join(args.celeb_hq_root, identity, file_id),
                "prompt": row['prompt'],
            })

        if args.resume:
            completed = load_completed_work(args.output_dir)
            print(f"Resuming: {len(completed)} repeats already completed")
        else:
            completed = set()
    else:
        # Random mode – build a non-deterministic plan inline (no resume).
        plan = []
        for _ in range(args.num_samples):
            fpath = random.choice(celeb_hq_image_paths)
            identity = fpath.split("/")[-2]
            file_id = os.path.basename(fpath)

            gender = gender_data.get(identity, "person").lower()
            if gender in ("man", "woman"):
                cfg.force_gender = gender
            else:
                cfg.force_gender = "person"

            prompt = generator.generate_one()['prompt']
            plan.append({
                "identity": identity,
                "file_id": file_id,
                "fpath": fpath,
                "prompt": prompt,
            })
        completed = set()

    num_samples = len(plan)

    # -----------------------------------------------------------------------
    # Main generation loop – iterate over the plan, generate images, and
    # keep only those that pass the ArcFace identity-similarity threshold.
    # -----------------------------------------------------------------------
    for i, sample in enumerate(plan):
        identity = sample["identity"]
        fid = sample["file_id"].split(".")[0]
        fpath = sample["fpath"]
        prompt = sample["prompt"]

        # Count how many repeats are already done for this sample.
        already_done = sum(
            1 for r in range(1, args.num_repeat + 1)
            if (identity, fid, r) in completed
        )
        if already_done >= args.num_repeat:
            print(f"Skipping sample {i+1}/{num_samples} ({identity}/{fid}) – all {args.num_repeat} repeats done")
            continue

        print(f"Processing identity: {identity}, file id: {fid}")
        print(f"Prompt: {prompt}")

        # For each sample we attempt up to `num_repeat` successful generations.
        # A generation only counts as successful if the ArcFace similarity
        # between the reference and generated face exceeds `score_thresh`.
        # We bail out after MAX_RETRIES total attempts to avoid infinite loops.
        num_generated = already_done + 1
        num_trials = 0
        while num_generated <= args.num_repeat and num_trials < MAX_RETRIES:
            print(f"Generating sample {i+1}/{num_samples}, repeat {num_generated}/{args.num_repeat}")
            num_trials += 1
            args.seed = random.randint(0, 0xFFFFFFFF)

            # Run the InfiniteYou pipeline to produce a single image.
            image = pipe(
                        id_image=Image.open(fpath).convert('RGB'),
                        prompt=prompt,
                        control_image=Image.open(args.control_image).convert('RGB') if args.control_image is not None else None,
                        seed=args.seed,
                        guidance_scale=args.guidance_scale,
                        num_steps=args.num_steps,
                        infusenet_conditioning_scale=args.infusenet_conditioning_scale,
                        infusenet_guidance_start=args.infusenet_guidance_start,
                        infusenet_guidance_end=args.infusenet_guidance_end,
                        cpu_offload=args.cpu_offload,
                        width=args.img_size[1],
                        height=args.img_size[0],
                    )

            # --- Save, score, and filter the generated image ---
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                output_path = os.path.join(args.output_dir, f"iden_{identity}_img_{fid}_sample_{i+1}_repeat_{num_generated}.png")
                image.save(output_path)
                print(f"Saved generated image to {output_path}")

                # Detect faces in both the original reference and the generated
                # image using ArcFace (InsightFace buffalo_l model).
                identity_img = cv2.imread(fpath)
                generated_img = cv2.imread(output_path)

                identity_faces = app.get(identity_img)
                generated_faces = app.get(generated_img)

                if len(identity_faces) > 0 and len(generated_faces) > 0:
                    # Compute cosine similarity between the two face embeddings.
                    # np.dot works here because the embeddings are L2-normalised.
                    identity_embedding = identity_faces[0].normed_embedding
                    generated_embedding = generated_faces[0].normed_embedding

                    similarity_score = np.dot(identity_embedding, generated_embedding)
                    print(f'The ArcFace similarity score is: {similarity_score}')

                    # Accept the image only if it exceeds the similarity threshold.
                    if similarity_score > args.score_thresh:
                        json_output_path = os.path.join(args.output_dir, "metadata.json")
                        metadata = {
                            "image_name": f"iden_{identity}_img_{fid}_sample_{i+1}_repeat_{num_generated}.png",
                            "prompt": prompt,
                            "identity": identity,
                            "file_id": os.path.basename(fpath),
                            "trial": i + 1,
                            "repeat": num_generated,
                            "similarity_score": float(similarity_score)
                        }

                        # Atomic write: write to temp file then os.replace to
                        # avoid corrupting metadata.json if killed mid-write.
                        atomic_json_append(json_output_path, metadata)
                        print(f"Saved metadata to {json_output_path}")
                        num_generated += 1
                else:
                    # No face detected in the generated image – discard it.
                    os.remove(output_path)

    print(f"Generation complete. Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()