# battle.py

import torch
from pathlib import Path
import logging
from PIL import Image
from methods.watermarked_diffusion_pipeline import WatermarkedDiffusionPipeline

class Battle:
    """Manages battles between Red and Blue teams"""

    def __init__(self, model_name="sdxl", optimize_memory=False, output_dir='outputs'):
        self.logger = self._setup_logging()
        self.model_name = model_name.lower()
        self.model = self._setup_model(optimize_memory=optimize_memory)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def _setup_model(self, optimize_memory=True):
        """Setup the Watermarked Diffusion Pipeline"""
        if self.model_name == "sdxl":
            # Load the SDXL base model
            pipe = WatermarkedDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16
            ).to("cuda")
            if optimize_memory:
                pipe.enable_attention_slicing()
                pipe.enable_model_cpu_offload()
            return pipe
        else:
            raise ValueError(f"Unknown model: {self.model_name}. Supported models: 'sdxl'")

    def run_battle(self, prompt: str, key: int = None):
        """Generate an image with watermark and attempt to detect the watermark"""

        self.logger.info(f"Prompt: {prompt}")
        self.logger.info(f"Using key: {key}")

        # Generate image using the diffusion model with watermark
        generated_image = self.model.generate(prompt, key=key)
        generated_image_path = self.output_dir / f"generated_image.png"
        generated_image.save(generated_image_path)
        self.logger.info(f"Generated image saved to {generated_image_path}")

        # Red Team attempts to remove watermark (students will implement this)
        # For now, we'll assume the attacked image is the same as the generated image
        attacked_image = generated_image  # Placeholder for attack

        # Blue Team attempts to detect the watermark
        extracted_key = self.model.detect(attacked_image)
        self.logger.info(f"Extracted key: {extracted_key}")

        detection_success = (extracted_key == key)
        self.logger.info(f"Watermark detection successful: {detection_success}")

        return {
            'generated_image': generated_image,
            'attacked_image': attacked_image,
            'extracted_key': extracted_key,
            'detection_success': detection_success
        }
