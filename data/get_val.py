import torch
from PIL import Image
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
def load_pipeline():
    # Constants for SDXL-Lightning
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    checkpoint = "sdxl_lightning_4step_unet.safetensors"
    
    # Initialize pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    
    # Set up scheduler
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        prediction_type="epsilon"
    )
    
    # Load the model weights
    pipe.unet.load_state_dict(
        load_file(
            hf_hub_download(repo, checkpoint),
            device="cuda"
        )
    )
    return pipe

pipe = load_pipeline()

# Path to the prompts file and reference images directory
prompts_file = "data/validation_prompts.txt"
reference_images_dir = Path("data/reference_images")
reference_images_dir.mkdir(parents=True, exist_ok=True)

# Read prompts from the file
with open(prompts_file, "r") as file:
    prompts = file.readlines()

# Generate images for each prompt and save
for i, prompt in enumerate(prompts):
    prompt = prompt.strip()  # Remove leading/trailing whitespace
    print(f"Generating image {i+1}/{len(prompts)} for prompt: {prompt}")
    
    # Generate image
    generated_image = pipe(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0,
        ).images[0]
    
    # Save the image
    image_path = reference_images_dir / f"generated_image_{i+1}.png"
    generated_image.save(image_path)

print("All reference images generated and saved!")
