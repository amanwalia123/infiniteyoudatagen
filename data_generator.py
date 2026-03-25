import argparse 
import csv
import json
import os
import ssl
import random
from glob import glob
import torch
from PIL import Image
import cv2
import numpy as np

# Calculate ArcFace Score
import insightface
from insightface.app import FaceAnalysis

from utils import get_ip_address, get_server_name
from pipelines.pipeline_infu_flux import InfUFluxPipeline
from prompt_generator import GenConfig, PromptGenerator

# Hugging face related SSL error fix
os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize a FaceAnalysis app with the ArcFace model
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(320, 320)) # Use ctx_id=0 for GPU, -1 for CPU


MAX_RETRIES = 10

def build_argparser():
    parser = argparse.ArgumentParser(description="Generate realistic data from celebrity faces and text prompts for realistic faces in the scenes")


    # Argument for CelebHQ dataset and prompt generation
    parser.add_argument("--celeb_hq_root", type=str, 
                        default="/netapp/output/aman.walia/data/CelebHQRefForRelease", 
                        help="Path to the directory containing CelebHQ image dirs")
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
    
    args = parser.parse_args()

    # Check arguments
    assert args.infu_flux_version == 'v1.0', 'Currently only supports InfiniteYou-FLUX v1.0'
    assert args.model_version in ['aes_stage2', 'sim_stage1'], 'Currently only supports model versions: aes_stage2 | sim_stage1'
    
    return args


def main():
    args = build_argparser()
    
    server_name = get_server_name()
    ip_addr = get_ip_address()
    
    # read all celeb hq image paths
    celeb_hq_image_paths = glob(f"{args.celeb_hq_root}/**/*.png", recursive=True)
    
    # read json file containing gender metadata
    with open(args.celeb_hq_gender_metadata) as f:
        gender_data = json.load(f)
        
    # Prompt configuration
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

    # Set cuda device
    torch.cuda.set_device(0)
    
    # Create output directory if it is not provided
    if args.output_dir is None:
        if server_name == "lambda":
            root = "/netapp/output/aman.walia/data/infiniteyou_face_dataset"
        elif server_name == "MLP":
            root = "/group-volume/TorAIC-Image-Quality/Aman-Contents/data/infiniteyou_face_dataset"
        elif server_name == "SPACE":
            root = "/group-volume/Aman-Contents/data/infiniteyou_face_dataset"
            
        device = args.cuda_device
        # ip_sub_address = ip_addr.split('.')[-1]
        
        args.output_dir = os.path.join(root, ip_addr, f"device_{device}")
    
    
    # Load pipeline
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
    # Load LoRAs (optional)
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
    
    # Perform inference
    if args.seed == 0:
        args.seed = torch.seed() & 0xFFFFFFFF

    # Prompt Generator
    generator = PromptGenerator(cfg)    

    if args.prompt_file is not None:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            prompt_metadata = list(reader)
        
        # Ignore num-samples when prompt_file is provided
        args.num_samples = len(prompt_metadata)
    else:
        prompt_metadata = None

    for i in range(args.num_samples):
        if prompt_metadata is not None:
            row = prompt_metadata[i]
            
            prompt = row['prompt']
            identity = row['identity']
            file_id = row['file_id']
            fid = file_id.split('.')[0]
            fpath = os.path.join(args.celeb_hq_root, identity, file_id)
        else:
            # select path from img paths
            fpath = random.choice(celeb_hq_image_paths)
            
            # extract gender from json metadata
            identity = fpath.split("/")[-2]
            fid = fpath.split("/")[-1].split(".")[0]  # file id
            
        print(f"Processing identity: {identity}, file id: {fid}")
        try:
            gender = gender_data[identity].lower()
        except KeyError:
            gender = "person"
            print(f"KeyError: identity '{identity}' not found, using default as person")
            
        # set the gender
        if gender in ["man", "woman"]:
            cfg.force_gender = gender

        if prompt_metadata is None:
            prompt = generator.generate_one()['prompt']
        print(f"Prompt: {prompt}")
        num_generated = 1
        num_trials = 0
        while(num_generated <= args.num_repeat and num_trials < MAX_RETRIES):
            
            print(f"Generating sample {i+1}/{args.num_samples}, repeat {num_generated}/{args.num_repeat}")
            num_trials += 1
            
            # randomize the seed
            args.seed = random.randint(0, 0xFFFFFFFF)

            image = pipe(
                        id_image=Image.open(fpath).convert('RGB'), # Load the image from the file path of the celeb hq image
                        prompt=prompt, # Text prompt generated by the PromptGenerator
                        control_image=Image.open(args.control_image).convert('RGB') if args.control_image is not None else None,
                        seed=args.seed,
                        guidance_scale=args.guidance_scale,
                        num_steps=args.num_steps,
                        infusenet_conditioning_scale=args.infusenet_conditioning_scale,
                        infusenet_guidance_start=args.infusenet_guidance_start,
                        infusenet_guidance_end=args.infusenet_guidance_end,
                        cpu_offload=args.cpu_offload,
                        width=args.img_size[1],  # Set the width of the generated image
                        height=args.img_size[0],  # Set the height of the generated image
                    )
            
            # Save the generated image
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                output_path = os.path.join(args.output_dir, f"iden_{identity}_img_{fid}_sample_{i+1}_repeat_{num_generated}.png")
                image.save(output_path)
                print(f"Saved generated image to {output_path}")
                
                # Load your images, which can be of different sizes
                identity_img = cv2.imread(fpath)
                generated_img = cv2.imread(output_path)

                # The `app.get()` function handles detection, alignment, and resizing internally
                identity_faces = app.get(identity_img)
                generated_faces = app.get(generated_img)

                if len(identity_faces) > 0 and len(generated_faces) > 0:
                    
                    # Extract the face embeddings from the normalized, standardized faces
                    identity_embedding = identity_faces[0].normed_embedding
                    generated_embedding = generated_faces[0].normed_embedding

                    # Calculate the cosine similarity
                    similarity_score = np.dot(identity_embedding, generated_embedding)

                    print(f'The ArcFace similarity score is: {similarity_score}')
                    if similarity_score > args.score_thresh:
                        # create a JSON file storing image name and prompt for all the generated images
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
                        
                        if os.path.exists(json_output_path):
                            with open(json_output_path, 'r') as f:
                                existing_data = json.load(f)
                        else:
                            existing_data = []

                        existing_data.append(metadata)

                        with open(json_output_path, 'w') as f:
                            json.dump(existing_data, f, indent=4)
                        print(f"Saved metadata to {json_output_path}")
                        
                        # update the number of generated images 
                        num_generated += 1
                else:
                    # delete the generated image
                    os.remove(output_path)
                    
if __name__ == "__main__":
    main()