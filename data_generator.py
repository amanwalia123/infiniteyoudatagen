import argparse
import json
import logging
import os
import ssl
import shutil

import random
from glob import glob
import torch
import torch.distributed as dist
from PIL import Image
import cv2
import numpy as np

# Calculate ArcFace Score
import insightface
from insightface.app import FaceAnalysis

from utils import get_ip_address, get_server_name
from pipelines.pipeline_infu_flux import InfUFluxPipeline
from prompt_generator2 import GenConfig, PromptGenerator

# Hugging face related SSL error fix
os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context

MAX_RETRIES = 10


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _get_logger(rank: int = 0) -> logging.Logger:
    """Return a logger prefixed with the current rank."""
    logger = logging.getLogger(f"datagen.rank{rank}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            f"[Rank {rank}] %(asctime)s %(levelname)s – %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Distributed initialisation / tear-down helpers
# ---------------------------------------------------------------------------

def init_distributed(rank: int, world_size: int,
                     master_addr: str, master_port: int,
                     backend: str = "nccl") -> bool:
    """Initialise torch.distributed process group.

    Returns True when the group was successfully created, False otherwise
    (e.g. when world_size == 1 and the caller chose not to use distributed).
    """
    if world_size <= 1:
        return False

    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))

    # Fall back to Gloo when NCCL is unavailable (CPU-only nodes / CI).
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )
    return True


def cleanup_distributed() -> None:
    """Destroy the distributed process group if it was initialised."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(logger: logging.Logger) -> None:
    """Execute a distributed barrier, logging the event."""
    if dist.is_available() and dist.is_initialized():
        logger.info("Reaching distributed barrier …")
        dist.barrier()
        logger.info("Passed distributed barrier.")

def build_argparser():
    parser = argparse.ArgumentParser(description="Generate realistic data from celeibrity faces and text prompts for realistic face in the scenes")


    # Argument for CelebHQ dataset and prompt generation
    parser.add_argument("--celeb_hq_root", type=str, 
                        default="/netapp/output/aman.walia/data/CelebHQRefForRelease", 
                        help="Path to the dierctory contaiing CelebHQ image dirs")
    parser.add_argument("--celeb_hq_gender_metadata", type=str, 
                        default="/netapp/output/aman.walia/data/CelebHQRefForRelease/gender_map.json", 
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
                        help="num of times to repeat wit same prompt")
    
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
    
    
    # Argument for infiniteyou
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

    # The LoRA options below are entirely optional. Here we provide two examples to facilitate users to try, but they are NOT used in our paper.
    parser.add_argument('--enable_realism_lora', action='store_true')
    parser.add_argument('--enable_anti_blur_lora', action='store_true')
    parser.add_argument('--enable_anti_blur_lora2', action='store_true')
    
    # Memory reduction options
    parser.add_argument('--quantize_8bit', action='store_true')
    parser.add_argument('--cpu_offload', action='store_true')

    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int, help="""seed (0 for random)""")

    # Argument for saving generated images
    parser.add_argument('--output_dir', type=str, default=None, help="Path to the output directory for generated images")

    # ---------------------------------------------------------------------------
    # Distributed / multi-GPU arguments
    # ---------------------------------------------------------------------------
    parser.add_argument('--distributed', action='store_true',
                        help="Enable torch.distributed for multi-GPU/multi-node data generation")
    parser.add_argument('--dist_backend', default='nccl', choices=['nccl', 'gloo'],
                        help="torch.distributed backend (default: nccl; use gloo for CPU-only nodes)")
    parser.add_argument('--dist_rank', type=int, default=0,
                        help="Global rank of this process in the distributed group")
    parser.add_argument('--dist_world_size', type=int, default=1,
                        help="Total number of processes in the distributed group")
    parser.add_argument('--dist_master_addr', type=str, default='127.0.0.1',
                        help="Hostname/IP of the master node for rendezvous")
    parser.add_argument('--dist_master_port', type=int, default=29500,
                        help="Port on the master node used for rendezvous")

    args = parser.parse_args()

    # Check arguments
    assert args.infu_flux_version == 'v1.0', 'Currently only supports InfiniteYou-FLUX v1.0'
    assert args.model_version in ['aes_stage2', 'sim_stage1'], 'Currently only supports model versions: aes_stage2 | sim_stage1'
    
    return args


def main():
    args = build_argparser()

    # ------------------------------------------------------------------
    # Distributed initialisation
    # ------------------------------------------------------------------
    rank = args.dist_rank
    world_size = args.dist_world_size
    is_distributed = False

    if args.distributed and world_size > 1:
        is_distributed = init_distributed(
            rank=rank,
            world_size=world_size,
            master_addr=args.dist_master_addr,
            master_port=args.dist_master_port,
            backend=args.dist_backend,
        )

    logger = _get_logger(rank)
    logger.info("Starting data generation (rank=%d / world_size=%d, distributed=%s)",
                rank, world_size, is_distributed)

    server_name = get_server_name()
    ip_addr = get_ip_address()

    # ------------------------------------------------------------------
    # Initialize FaceAnalysis here (after potential CUDA device selection)
    # ------------------------------------------------------------------
    if torch.cuda.is_available() and args.cuda_device < torch.cuda.device_count():
        torch.cuda.set_device(args.cuda_device)
        face_ctx_id = args.cuda_device
    else:
        face_ctx_id = -1  # CPU fallback for insightface
    face_app = FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=face_ctx_id, det_size=(320, 320))

    # ------------------------------------------------------------------
    # Read all celeb hq image paths
    # ------------------------------------------------------------------
    celeb_hq_image_paths = glob(f"{args.celeb_hq_root}/**/*.png", recursive=True)
    if not celeb_hq_image_paths:
        logger.error("No images found in %s – aborting.", args.celeb_hq_root)
        cleanup_distributed()
        raise FileNotFoundError(f"No PNG images found under {args.celeb_hq_root}")

    # ------------------------------------------------------------------
    # Partition the image list across all ranks so that every GPU works on
    # a unique, non-overlapping subset.  Each rank handles every world_size-th
    # image starting from its own rank index.
    # ------------------------------------------------------------------
    if world_size > 1:
        celeb_hq_image_paths = celeb_hq_image_paths[rank::world_size]
        logger.info("This rank will sample from %d images (out of total pool).",
                    len(celeb_hq_image_paths))

    # read json file containing gender metadata
    with open(args.celeb_hq_gender_metadata) as f:
        gender_data = json.load(f)

    # ------------------------------------------------------------------
    # Prompt configuration
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------------
    if args.output_dir is None:
        if server_name == "lambda":
            root = "/netapp/output/aman.walia/data/infiniteyou_face_dataset"
        elif server_name == "MLP":
            root = "/group-volume/TorAIC-Image-Quality/Aman-Contents/data/infiniteyou_face_dataset"
        elif server_name == "SPACE":
            root = "/group-volume/Aman-Contents/data/infiniteyou_face_dataset"
        else:
            root = "/tmp/infiniteyou_face_dataset"

        device = args.cuda_device
        args.output_dir = os.path.join(root, ip_addr, f"device_{device}")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Output directory: %s", args.output_dir)

    # ------------------------------------------------------------------
    # Barrier: wait until ALL ranks have created their output directories
    # before loading heavy models (prevents I/O races on shared filesystems).
    # ------------------------------------------------------------------
    barrier(logger)

    # ------------------------------------------------------------------
    # Load pipeline
    # ------------------------------------------------------------------
    infu_model_path = os.path.join(args.model_dir, f'infu_flux_{args.infu_flux_version}', args.model_version)
    insightface_root_path = os.path.join(args.model_dir, 'supports', 'insightface')
    logger.info("Loading InfUFluxPipeline from %s …", infu_model_path)
    pipe = InfUFluxPipeline(
        base_model_path=args.base_model_path,
        infu_model_path=infu_model_path,
        insightface_root_path=insightface_root_path,
        infu_flux_version=args.infu_flux_version,
        model_version=args.model_version,
        quantize_8bit=args.quantize_8bit,
        cpu_offload=args.cpu_offload,
    )
    # Load LoRAs (optional)
    lora_dir = os.path.join(args.model_dir, 'supports', 'optional_loras')
    if not os.path.exists(lora_dir):
        lora_dir = '/group-volume/Aman-Contents/InfiniteYou/supports/optional_loras'
    loras = []

    if args.enable_realism_lora:
        loras.append([os.path.join(lora_dir, 'flux_realism_lora.safetensors'), 'realism', 1.0])

    if args.enable_anti_blur_lora:
        loras.append([os.path.join(lora_dir, 'flux_anti_blur_lora.safetensors'), 'anti_blur', 1.0])

    if args.enable_anti_blur_lora2:
        loras.append([os.path.join(lora_dir, 'FLUX-dev-lora-AntiBlur.safetensors'), 'anti_blur', 3.0])

    pipe.load_loras(loras)

    # ------------------------------------------------------------------
    # Barrier: all ranks have finished loading models – begin generation
    # ------------------------------------------------------------------
    barrier(logger)
    logger.info("All ranks ready – starting generation loop.")

    # Perform inference
    if args.seed == 0:
        args.seed = torch.seed() & 0xFFFFFFFF

    # Prompt Generator
    generator = PromptGenerator(cfg)

    for i in range(args.num_samples):
        if not celeb_hq_image_paths:
            logger.warning("No images available for this rank – skipping iteration %d.", i)
            continue

        # select path from img paths assigned to this rank
        fpath = random.choice(celeb_hq_image_paths)

        # extract gender from json metadata
        identity = fpath.split("/")[-2]
        fid = fpath.split("/")[-1].split(".")[0]  # file id
        logger.info("Processing identity: %s, file id: %s", identity, fid)
        try:
            gender = gender_data[identity].lower()
        except KeyError:
            gender = "person"
            logger.warning("KeyError: identity '%s' not in gender metadata – using 'person'.", identity)

        # set the gender
        if gender in ["man", "woman"]:
            cfg.force_gender = gender

        prompt = generator.generate_one()['prompt']
        logger.info("Prompt: %s", prompt)
        num_generated = 1
        num_trials = 0
        while num_generated <= args.num_repeat and num_trials < MAX_RETRIES:

            logger.info("Generating sample %d/%d, repeat %d/%d",
                        i + 1, args.num_samples, num_generated, args.num_repeat)
            num_trials += 1

            # randomize the seed
            args.seed = random.randint(0, 0xFFFFFFFF)

            try:
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
            except Exception as exc:
                logger.error("Pipeline call failed (trial %d): %s", num_trials, exc)
                continue

            # Save the generated image
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(
                args.output_dir,
                f"iden_{identity}_img_{fid}_sample_{i+1}_repeat_{num_generated}.png"
            )
            image.save(output_path)
            logger.info("Saved generated image to %s", output_path)

            # ArcFace similarity check
            identity_img = cv2.imread(fpath)
            generated_img = cv2.imread(output_path)

            identity_faces = face_app.get(identity_img)
            generated_faces = face_app.get(generated_img)

            if len(identity_faces) > 0 and len(generated_faces) > 0:
                identity_embedding = identity_faces[0].normed_embedding
                generated_embedding = generated_faces[0].normed_embedding
                similarity_score = float(np.dot(identity_embedding, generated_embedding))

                logger.info("ArcFace similarity score: %.4f", similarity_score)
                if similarity_score > args.score_thresh:
                    # Each rank writes to its own metadata file to avoid
                    # concurrent write races on a shared filesystem.
                    json_output_path = os.path.join(
                        args.output_dir, f"metadata_rank{rank}.json"
                    )
                    metadata = {
                        "image_name": os.path.basename(output_path),
                        "prompt": prompt,
                        "identity": identity,
                        "file_id": os.path.basename(fpath),
                        "trial": i + 1,
                        "repeat": num_generated,
                        "similarity_score": similarity_score,
                        "rank": rank,
                    }

                    if os.path.exists(json_output_path):
                        with open(json_output_path, 'r') as f:
                            existing_data = json.load(f)
                    else:
                        existing_data = []

                    existing_data.append(metadata)

                    with open(json_output_path, 'w') as f:
                        json.dump(existing_data, f, indent=4)
                    logger.info("Saved metadata to %s", json_output_path)

                    # Update the number of generated images
                    num_generated += 1
                else:
                    # Similarity below threshold – discard the image
                    os.remove(output_path)
                    logger.info("Discarded image (similarity %.4f < threshold %.4f).",
                                similarity_score, args.score_thresh)
            else:
                # No face detected in one or both images – discard
                os.remove(output_path)
                logger.warning("No face detected in identity or generated image – discarding.")

    # ------------------------------------------------------------------
    # Final barrier: wait for all ranks to finish before teardown
    # ------------------------------------------------------------------
    barrier(logger)
    logger.info("Generation complete for rank %d.", rank)
    cleanup_distributed()
                    
if __name__ == "__main__":
    main()