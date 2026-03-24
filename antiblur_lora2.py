import torch
from diffusers import FluxPipeline

import os
os.environ['CURL_CA_BUNDLE'] = ''

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

LORA_SCALE = 1.5

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur", weight_name="FLUX-dev-lora-AntiBlur.safetensors")
pipe.fuse_lora(lora_scale=3.0)
pipe.to("cuda")

prompt = "a young college student, walking on the street, campus background, photography with crystal clear in focus background"
# prompt = "photorealistic photography of Cafe environment, with young 18 years old waitress, blonde hair, in stylist uniform."


image = pipe(prompt, 
             num_inference_steps=30, 
             guidance_scale=3.5,
             width=768, height=1024,
            ).images[0]
image.save(f"example_2_{LORA_SCALE}.png")
