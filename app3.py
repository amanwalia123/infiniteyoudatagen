import argparse
import os
import random
import sys

import gradio as gr
import torch
from PIL import Image

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Assuming the pipeline is available in your environment
# from pipelines.pipeline_infu_flux import InfUFluxPipeline

import os
os.environ['CURL_CA_BUNDLE'] = ''

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# A dummy class to mock the pipeline for demonstration purposes if needed
from pipelines.pipeline_infu_flux import InfUFluxPipeline

def preprocess_face(image, app, size=(112, 112)):
    """
    Detect, align, and preprocess the face for the ArcFace model.
    - Detects the face using the insightface app.
    - Aligns and resizes the face to the required size.
    - Normalizes pixel values to [-1, 1].
    """
    faces = app.get(image)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    
    # Use the first detected face
    face = faces[0]
    aligned_face = face['bbox']  # Aligned face bounding box
    cropped_face = image[int(aligned_face[1]):int(aligned_face[3]), int(aligned_face[0]):int(aligned_face[2])]
    
    # Resize and normalize
    cropped_face = cv2.resize(cropped_face, size)
    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    cropped_face = (cropped_face / 127.5) - 1.0  # Normalize to [-1, 1]
    return np.transpose(cropped_face, (2, 0, 1))  # HWC to CHW

def calculate_arcface_score(detected_face, generated_face):
    """
    Calculate the ArcFace score (cosine similarity) between two face images.
    """
    # Load the ArcFace model
    app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Preprocess the images
    detected_face = preprocess_face(detected_face, app)
    generated_face = preprocess_face(generated_face, app)

    # Extract embeddings
    detected_embedding = app.model.get_embedding(detected_face)
    generated_embedding = app.model.get_embedding(generated_face)

    # Calculate cosine similarity
    score = cosine_similarity([detected_embedding], [generated_embedding])[0][0]
    return score


# The core function for inference, adapted for Gradio
def run_inference(
    id_image,
    control_image,
    prompt,
    infu_flux_version,
    model_version,
    cuda_device,
    seed,
    guidance_scale,
    num_steps,
    infusenet_conditioning_scale,
    infusenet_guidance_start,
    infusenet_guidance_end,
    enable_realism_lora,
    enable_anti_blur_lora,
    enable_anti_blur_lora2,
    quantize_8bit,
    cpu_offload
):
    # This section can reuse the logic from your original script
    # with the function arguments replacing the argparse `args` object.

    # Check arguments
    assert infu_flux_version == 'v1.0', 'Currently only supports InfiniteYou-FLUX v1.0'
    assert model_version in ['aes_stage2', 'sim_stage1'], 'Currently only supports model versions: aes_stage2 | sim_stage1'

    # Set cuda device (Gradio runs on the server, so device management is server-side)
    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device)

    # Load pipeline
    base_model_path = 'black-forest-labs/FLUX.1-dev'
    model_dir = 'ByteDance/InfiniteYou'
    infu_model_path = os.path.join(model_dir, f'infu_flux_{infu_flux_version}', model_version)
    insightface_root_path = os.path.join(model_dir, 'supports', 'insightface')
    
    pipe = InfUFluxPipeline(
        base_model_path=base_model_path,
        infu_model_path=infu_model_path,
        insightface_root_path=insightface_root_path,
        infu_flux_version=infu_flux_version,
        model_version=model_version,
        quantize_8bit=quantize_8bit,
        cpu_offload=cpu_offload,
    )

    # Load LoRAs (optional)
    lora_dir = os.path.join(model_dir, 'supports', 'optional_loras')
    if not os.path.exists(lora_dir):
        lora_dir = './models/InfiniteYou/supports/optional_loras'
    
    loras = []
    if enable_realism_lora:
        loras.append([os.path.join(lora_dir, 'flux_realism_lora.safetensors'), 'realism', 1.0])
    if enable_anti_blur_lora:
        loras.append([os.path.join(lora_dir, 'flux_anti_blur_lora.safetensors'), 'anti_blur', 1.0])
    if enable_anti_blur_lora2:
        loras.append([os.path.join(lora_dir, 'FLUX-dev-lora-AntiBlur.safetensors'), 'anti_blur', 3.0])
    
    
    pipe.load_loras(loras)
    
    # Handle seed
    if seed == 0:
        seed = random.randint(0, 0xFFFFFFFF)

    # Convert control_image for pipeline, if provided
    control_image_pil = Image.open(control_image).convert('RGB') if control_image else None
    
    # Perform inference
    image = pipe(
        id_image=id_image, # Gradio passes PIL images directly
        prompt=prompt,
        control_image=control_image_pil,
        seed=seed,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
        infusenet_conditioning_scale=infusenet_conditioning_scale,
        infusenet_guidance_start=infusenet_guidance_start,
        infusenet_guidance_end=infusenet_guidance_end,
        cpu_offload=cpu_offload,
    )
    
    return image

# Build the Gradio interface
with gr.Blocks(title="InfiniteYou-FLUX") as demo:
    gr.Markdown("# InfiniteYou-FLUX Web Interface")
    gr.Markdown("Upload an ID image and an optional control image, then provide a prompt to generate a new image.")

    with gr.Row():
        with gr.Column():
            id_image = gr.Image(label="ID Image", type="pil", sources=["upload", "clipboard"])
            control_image = gr.Image(label="Control Image (Optional)", type="pil", sources=["upload", "clipboard"])

        with gr.Column():
            output_image = gr.Image(label="Generated Image", type="pil")

    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            value='A man, portrait, cinematic',
            placeholder="A man, portrait, cinematic",
        )
    
    with gr.Accordion("Advanced Options", open=False):
        with gr.Row():
            with gr.Column():
                infu_flux_version = gr.Dropdown(
                    label="InfiniteYou-FLUX Version",
                    choices=['v1.0'],
                    value='v1.0',
                )
                model_version = gr.Dropdown(
                    label="Model Version",
                    choices=['aes_stage2', 'sim_stage1'],
                    value='aes_stage2',
                )
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=10.0,
                    step=0.1,
                    value=3.5,
                )
                num_steps = gr.Slider(
                    label="Number of Steps",
                    minimum=10,
                    maximum=50,
                    step=1,
                    value=30,
                )

            with gr.Column():
                infusenet_conditioning_scale = gr.Slider(
                    label="InfuseNet Conditioning Scale",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                )
                infusenet_guidance_start = gr.Slider(
                    label="InfuseNet Guidance Start",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.0,
                )
                infusenet_guidance_end = gr.Slider(
                    label="InfuseNet Guidance End",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=1.0,
                )
                cuda_device = gr.Number(
                    label="CUDA Device",
                    value=0,
                    precision=0
                )
                seed = gr.Number(
                    label="Seed (0 for random)",
                    value=0,
                    precision=0
                )

        with gr.Row():
            enable_realism_lora = gr.Checkbox(label="Enable Realism LoRA", value=False)
            enable_anti_blur_lora = gr.Checkbox(label="Enable Anti-Blur LoRA", value=False)
            enable_anti_blur_lora2 = gr.Checkbox(label="Enable Anti-Blur LoRA from Shakker-Labs", value=False)
            quantize_8bit = gr.Checkbox(label="Quantize to 8-bit", value=False)
            cpu_offload = gr.Checkbox(label="CPU Offload", value=False)

    generate_btn = gr.Button("Generate", variant="primary")
    generate_btn.click(
        fn=run_inference,
        inputs=[
            id_image,
            control_image,
            prompt,
            infu_flux_version,
            model_version,
            cuda_device,
            seed,
            guidance_scale,
            num_steps,
            infusenet_conditioning_scale,
            infusenet_guidance_start,
            infusenet_guidance_end,
            enable_realism_lora,
            enable_anti_blur_lora,
            enable_anti_blur_lora2,
            quantize_8bit,
            cpu_offload
        ],
        outputs=[output_image]
    )
    score_btn = gr.Button("ArcFace Score", variant="primary")
    score_btn.click(
        fn=
    )

if __name__ == "__main__":
    demo.launch()