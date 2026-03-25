#!/usr/bin/env python3
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

Multi-instance / multi-GPU mode:
  - Requires --prompt_file
  - Prompts are partitioned globally across all workers in all instances
  - Cluster topology is provided explicitly via:
      --cluster_layout "4,4,2"
    meaning:
      instance 0 has 4 GPUs
      instance 1 has 4 GPUs
      instance 2 has 2 GPUs
  - Each worker is uniquely identified by:
      global_rank = sum(cluster_layout[:instance_rank]) + cuda_device
      world_size  = sum(cluster_layout)

Examples:
    python data_generator.py --output_dir ./output --num-samples 100

    python data_generator.py \
        --prompt_file splits/train_metadata.csv \
        --cluster_layout "4,4,2" \
        --instance_rank 1 \
        --cuda_device 2 \
        --resume

Important runtime assumption:
  - Each process should be launched with CUDA_VISIBLE_DEVICES=<local_gpu>
  - Inside the process, the chosen GPU becomes torch device 0
"""

import argparse
import csv
import json
import os
import ssl
import random
import tempfile
from glob import glob
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from insightface.app import FaceAnalysis

from utils import get_ip_address, get_server_name
from pipelines.pipeline_infu_flux import InfUFluxPipeline
from prompt_generator import GenConfig, PromptGenerator

# ---------------------------------------------------------------------------
# Global SSL workaround – required for downloading HuggingFace model weights
# in environments with custom/broken certificate chains.
# ---------------------------------------------------------------------------
os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

# Maximum number of generation attempts per prompt/repeat before giving up.
MAX_RETRIES = 10


def parse_cluster_layout(layout_str: str) -> List[int]:
    """
    Parse a layout string like '4,4,2' into [4, 4, 2].
    """
    try:
        layout = [int(x.strip()) for x in layout_str.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid --cluster_layout '{layout_str}'. "
            "Expected comma-separated integers, e.g. '4,4,2'."
        ) from e

    if not layout:
        raise argparse.ArgumentTypeError(
            "--cluster_layout must contain at least one integer."
        )

    if any(x <= 0 for x in layout):
        raise argparse.ArgumentTypeError(
            "--cluster_layout must contain only positive integers."
        )

    return layout


def atomic_json_append(json_path: str, entry: dict) -> None:
    """
    Append `entry` to the JSON array at `json_path` using an atomic write.
    """
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    dir_name = os.path.dirname(json_path)
    os.makedirs(dir_name, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        os.replace(tmp_path, json_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_completed_work(output_dir: str) -> set:
    """
    Read metadata.json and return the set of completed work items.

    Each item is stored as:
        (identity, file_id_stem, repeat)
    """
    json_output_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(json_output_path):
        return set()

    with open(json_output_path, "r", encoding="utf-8") as f:
        existing_data = json.load(f)

    completed = set()
    for entry in existing_data:
        completed.add(
            (entry["identity"], entry["file_id"].split(".")[0], entry["repeat"])
        )
    return completed


def build_argparser() -> argparse.Namespace:
    """
    Define and parse all CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate realistic data from celebrity faces and text prompts "
            "for realistic faces in scenes."
        )
    )

    # --- CelebHQ dataset & prompt generation arguments ---
    parser.add_argument(
        "--celeb_hq_root",
        type=str,
        default="/netapp/output/aman.walia/data/CelebHQRefForRelease",
        help="Path to the directory containing CelebHQ image dirs.",
    )
    parser.add_argument(
        "--celeb_hq_gender_metadata",
        type=str,
        default="gender_map.json",
        help="Path to the gender metadata JSON file.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of dataset samples to generate (used only in random mode).",
    )
    parser.add_argument(
        "--num-repeat",
        type=int,
        default=3,
        help="Number of times to repeat with the same prompt.",
    )
    parser.add_argument("--force-gender", type=str, choices=["man", "woman"])
    parser.add_argument("--score_thresh", type=float, default=0.45)
    parser.add_argument(
        "--flux-positive-focus",
        "--flux",
        action="store_true",
        help="Use positive-only deep-focus phrasing for Flux Dev.",
    )
    parser.add_argument(
        "--scene-packs-file",
        type=str,
        default=None,
        help="(Legacy) combined JSON with multiple packs.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to CSV containing prompt, identity, and file_id.",
    )

    # --- InfiniteYou model & inference arguments ---
    parser.add_argument("--base_model_path", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--model_dir", default="./models")
    parser.add_argument("--control_image", default=None, help="Control image [optional].")
    parser.add_argument(
        "--infu_flux_version",
        default="v1.0",
        help="InfiniteYou-FLUX version: currently only v1.0.",
    )
    parser.add_argument(
        "--model_version",
        default="aes_stage2",
        help="Model version: aes_stage2 | sim_stage1.",
    )
    parser.add_argument("--guidance_scale", default=3.5, type=float)
    parser.add_argument("--num_steps", default=30, type=int)
    parser.add_argument("--infusenet_conditioning_scale", default=1.0, type=float)
    parser.add_argument("--infusenet_guidance_start", default=0.0, type=float)
    parser.add_argument("--infusenet_guidance_end", default=1.0, type=float)
    parser.add_argument(
        "--img_size",
        default=(1152, 1024),
        type=int,
        nargs=2,
        help="Image size for the generated images (width, height).",
    )

    # --- Optional LoRA adapters ---
    parser.add_argument("--enable_realism_lora", action="store_true")
    parser.add_argument("--enable_anti_blur_lora", action="store_true")
    parser.add_argument("--enable_anti_blur_lora2", action="store_true")

    # --- Memory-reduction options ---
    parser.add_argument("--quantize_8bit", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true")

    # --- Worker / device args ---
    parser.add_argument(
        "--cuda_device",
        default=0,
        type=int,
        help=(
            "Local GPU index on this instance. "
            "This is used for worker identity and sharding."
        ),
    )
    parser.add_argument("--seed", default=0, type=int, help="Seed (0 for random).")
    parser.add_argument(
        "--cluster_layout",
        type=parse_cluster_layout,
        default=[1],
        help="Comma-separated GPU counts per instance, e.g. '4,4,2'.",
    )
    parser.add_argument(
        "--instance_rank",
        type=int,
        default=0,
        help="Rank of this cluster instance/node in [0, len(cluster_layout)-1].",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the output directory for generated images.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a previously interrupted run. Requires --prompt_file. "
            "Skips samples/repeats already present in metadata.json."
        ),
    )

    args = parser.parse_args()

    assert args.infu_flux_version == "v1.0", (
        "Currently only supports InfiniteYou-FLUX v1.0"
    )
    assert args.model_version in ["aes_stage2", "sim_stage1"], (
        "Currently only supports model versions: aes_stage2 | sim_stage1"
    )

    if args.resume and args.prompt_file is None:
        parser.error(
            "--resume requires --prompt_file "
            "(resume is only supported with a deterministic prompt CSV)."
        )

    world_size = sum(args.cluster_layout)
    num_instances = len(args.cluster_layout)

    if args.instance_rank < 0 or args.instance_rank >= num_instances:
        parser.error("--instance_rank must be in [0, len(cluster_layout)-1].")

    local_gpu_count = args.cluster_layout[args.instance_rank]

    if args.cuda_device < 0 or args.cuda_device >= local_gpu_count:
        parser.error(
            f"--cuda_device must be in [0, {local_gpu_count - 1}] "
            f"for instance_rank={args.instance_rank}."
        )

    if world_size > 1 and args.prompt_file is None:
        parser.error(
            "Multi-worker execution requires --prompt_file "
            "for deterministic work partitioning."
        )

    return args


def get_default_output_dir(args: argparse.Namespace, server_name: str, ip_addr: str) -> str:
    """
    Derive a default output directory when none is explicitly provided.
    """
    if server_name == "lambda":
        root = "/netapp/output/aman.walia/data/infiniteyou_face_dataset2"
    elif server_name == "MLP":
        root = "/group-volume/TorAIC-Image-Quality/Aman-Contents/data/infiniteyou_face_dataset"
    elif server_name == "SPACE":
        root = "/group-volume/Aman-Contents/data/infiniteyou_face_dataset"
    else:
        root = "./output"

    return os.path.join(
        root,
        f"instance_{args.instance_rank}",
        ip_addr,
        f"device_{args.cuda_device}",
    )


def build_prompt_plan_random(
    args: argparse.Namespace,
    celeb_hq_image_paths: List[str],
    gender_data: dict,
    generator: PromptGenerator,
    cfg: GenConfig,
) -> list:
    """
    Random, non-deterministic prompt plan for single-worker mode.
    """
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

        prompt = generator.generate_one()["prompt"]

        plan.append(
            {
                "identity": identity,
                "file_id": file_id,
                "fpath": fpath,
                "prompt": prompt,
            }
        )

    return plan


def build_prompt_plan_from_csv(args: argparse.Namespace) -> list:
    """
    Deterministically shard CSV rows across all workers in all instances.
    """
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    world_size = sum(args.cluster_layout)
    local_gpu_count = args.cluster_layout[args.instance_rank]
    global_rank = sum(args.cluster_layout[:args.instance_rank]) + args.cuda_device

    if args.cuda_device >= local_gpu_count:
        raise ValueError(
            f"Invalid cuda_device={args.cuda_device} for instance_rank={args.instance_rank}. "
            f"This instance has {local_gpu_count} GPUs according to "
            f"cluster_layout={args.cluster_layout}."
        )

    my_rows = all_rows[global_rank::world_size]

    print(
        f"cluster_layout={args.cluster_layout}, "
        f"instance_rank={args.instance_rank}, "
        f"local_gpu={args.cuda_device}/{local_gpu_count}, "
        f"global_rank={global_rank}/{world_size}: "
        f"assigned {len(my_rows)}/{len(all_rows)} rows"
    )

    plan = []
    for row in my_rows:
        identity = row["identity"]
        file_id = row["file_id"]
        plan.append(
            {
                "identity": identity,
                "file_id": file_id,
                "fpath": os.path.join(args.celeb_hq_root, identity, file_id),
                "prompt": row["prompt"],
            }
        )

    return plan


def main() -> None:
    """
    Entry point: parse args, load models, generate images, and filter by identity score.
    """
    args = build_argparser()

    server_name = get_server_name()
    ip_addr = get_ip_address()

    celeb_hq_image_paths = glob(f"{args.celeb_hq_root}/**/*.png", recursive=True)

    with open(args.celeb_hq_gender_metadata, "r", encoding="utf-8") as f:
        gender_data = json.load(f)

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

    # Runtime assumption:
    # each worker process is launched with CUDA_VISIBLE_DEVICES=<local_gpu>.
    # Inside the process, that GPU becomes visible as device 0.
    torch.cuda.set_device(0)

    if args.output_dir is None:
        args.output_dir = get_default_output_dir(args, server_name, ip_addr)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load ArcFace after CUDA device is configured.
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(320, 320))

    # -----------------------------------------------------------------------
    # Load InfiniteYou-FLUX pipeline.
    # -----------------------------------------------------------------------
    infu_model_path = os.path.join(
        args.model_dir,
        f"infu_flux_{args.infu_flux_version}",
        args.model_version,
    )
    insightface_root_path = os.path.join(args.model_dir, "supports", "insightface")

    pipe = InfUFluxPipeline(
        base_model_path=args.base_model_path,
        infu_model_path=infu_model_path,
        insightface_root_path=insightface_root_path,
        infu_flux_version=args.infu_flux_version,
        model_version=args.model_version,
        quantize_8bit=args.quantize_8bit,
        cpu_offload=args.cpu_offload,
    )

    # Optionally attach LoRA adapters.
    lora_dir = os.path.join(args.model_dir, "supports", "optional_loras")

    loras = []
    if args.enable_realism_lora:
        loras.append(
            [os.path.join(lora_dir, "flux_realism_lora.safetensors"), "realism", 1.0]
        )
    if args.enable_anti_blur_lora:
        loras.append(
            [os.path.join(lora_dir, "flux_anti_blur_lora.safetensors"), "anti_blur", 1.0]
        )
    if args.enable_anti_blur_lora2:
        loras.append(
            [os.path.join(lora_dir, "FLUX-dev-lora-AntiBlur.safetensors"), "anti_blur", 3.0]
        )

    pipe.load_loras(loras)

    if args.seed == 0:
        args.seed = torch.seed() & 0xFFFFFFFF

    generator = PromptGenerator(cfg)

    # -----------------------------------------------------------------------
    # Build the work list for this worker.
    # -----------------------------------------------------------------------
    if args.prompt_file is not None:
        plan = build_prompt_plan_from_csv(args)
        if args.resume:
            completed = load_completed_work(args.output_dir)
            print(f"Resuming: {len(completed)} repeats already completed in {args.output_dir}")
        else:
            completed = set()
    else:
        plan = build_prompt_plan_random(
            args=args,
            celeb_hq_image_paths=celeb_hq_image_paths,
            gender_data=gender_data,
            generator=generator,
            cfg=cfg,
        )
        completed = set()

    num_samples = len(plan)

    # -----------------------------------------------------------------------
    # Main generation loop.
    # -----------------------------------------------------------------------
    for i, sample in enumerate(plan):
        identity = sample["identity"]
        fid = sample["file_id"].split(".")[0]
        fpath = sample["fpath"]
        prompt = sample["prompt"]

        if not os.path.exists(fpath):
            print(f"Skipping missing reference image: {fpath}")
            continue

        already_done = sum(
            1
            for r in range(1, args.num_repeat + 1)
            if (identity, fid, r) in completed
        )

        if already_done >= args.num_repeat:
            print(
                f"Skipping sample {i + 1}/{num_samples} ({identity}/{fid}) "
                f"– all {args.num_repeat} repeats done"
            )
            continue

        print(f"Processing identity: {identity}, file id: {fid}")
        print(f"Prompt: {prompt}")

        num_generated = already_done + 1
        num_trials = 0

        while num_generated <= args.num_repeat and num_trials < MAX_RETRIES:
            print(
                f"Generating sample {i + 1}/{num_samples}, "
                f"repeat {num_generated}/{args.num_repeat}"
            )
            num_trials += 1
            args.seed = random.randint(0, 0xFFFFFFFF)

            image = pipe(
                id_image=Image.open(fpath).convert("RGB"),
                prompt=prompt,
                control_image=(
                    Image.open(args.control_image).convert("RGB")
                    if args.control_image is not None
                    else None
                ),
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

            output_path = os.path.join(
                args.output_dir,
                f"iden_{identity}_img_{fid}_sample_{i + 1}_repeat_{num_generated}.png",
            )
            image.save(output_path)
            print(f"Saved generated image to {output_path}")

            identity_img = cv2.imread(fpath)
            generated_img = cv2.imread(output_path)

            if identity_img is None:
                print(f"Failed to read identity image: {fpath}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                continue

            if generated_img is None:
                print(f"Failed to read generated image: {output_path}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                continue

            identity_faces = app.get(identity_img)
            generated_faces = app.get(generated_img)

            if len(identity_faces) > 0 and len(generated_faces) > 0:
                identity_embedding = identity_faces[0].normed_embedding
                generated_embedding = generated_faces[0].normed_embedding

                similarity_score = float(np.dot(identity_embedding, generated_embedding))
                print(f"The ArcFace similarity score is: {similarity_score}")

                if similarity_score > args.score_thresh:
                    json_output_path = os.path.join(args.output_dir, "metadata.json")
                    metadata = {
                        "image_name": f"iden_{identity}_img_{fid}_sample_{i + 1}_repeat_{num_generated}.png",
                        "prompt": prompt,
                        "identity": identity,
                        "file_id": os.path.basename(fpath),
                        "trial": i + 1,
                        "repeat": num_generated,
                        "similarity_score": similarity_score,
                        "instance_rank": args.instance_rank,
                        "cuda_device": args.cuda_device,
                        "cluster_layout": args.cluster_layout,
                        "local_gpu_count": args.cluster_layout[args.instance_rank],
                    }

                    atomic_json_append(json_output_path, metadata)
                    completed.add((identity, fid, num_generated))
                    print(f"Saved metadata to {json_output_path}")
                    num_generated += 1
                else:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                        print(f"Removed low-score image: {output_path}")
            else:
                if os.path.exists(output_path):
                    os.remove(output_path)
                    print(f"Removed image with no detected face: {output_path}")

        if num_generated <= args.num_repeat:
            print(
                f"Warning: exhausted MAX_RETRIES={MAX_RETRIES} for "
                f"{identity}/{fid}; completed repeats up to {num_generated - 1}"
            )

    print(f"Generation complete. Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()