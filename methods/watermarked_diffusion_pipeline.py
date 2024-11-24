# methods/watermarked_diffusion_pipeline.py

import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
import numpy as np

class BaseWatermarkedDiffusionPipeline:
    def __init__(self, device: str = "cuda"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_pipeline()

    def load_pipeline(self):
        # Constants for SDXL-Lightning
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        checkpoint = "sdxl_lightning_4step_unet.safetensors"
        
        # Initialize pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(self.device)
        
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
                device=self.device
            )
        )
        return pipe

    def generate(self, prompt: str, key: int = None, **generate_kwargs) -> Image.Image:
        """
        Generates an image from the prompt, embedding a watermark if a key is provided.

        Args:
            prompt (str): The text prompt for image generation.
            key (int, optional): An integer key used to embed the watermark.

        Returns:
            Image: Generated (and potentially watermarked) image.
        """
        # Generate image using the customized model
        image = self.model(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0,
            **generate_kwargs
        ).images[0]

        return image


    def detect(self, image: Image.Image) -> int:
        # Simple example of watermark detection (students should improve this)
        """
        Detects a watermark in an image.
        return a random integer from 1 to 100 for the baseline
        """
        return np.random.randint(1, 101)
