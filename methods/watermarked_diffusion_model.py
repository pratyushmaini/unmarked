# methods/watermarked_diffusion_pipeline.py

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import numpy as np

class WatermarkedDiffusionPipeline(StableDiffusionXLPipeline):
    """
    A diffusion pipeline that can embed a watermark into generated images
    and detect the watermark from images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def generate(self, prompt: str, key: int = None, **generate_kwargs) -> Image:
        """
        Generates an image from the prompt, embedding a watermark if a key is provided.

        Args:
            prompt (str): The text prompt for image generation.
            key (int, optional): An integer key used to embed the watermark.

        Returns:
            Image: Generated (and watermarked) image.
        """
        # Generate image using the base pipeline
        image = super().__call__(prompt=prompt, **generate_kwargs).images[0]
        
        if key is not None:
            # Embed the watermark using the key
            image = self.embed_watermark(image, key)
        
        return image
    
    def detect(self, image: Image) -> int:
        """
        Attempts to extract the watermark key from the image.

        Args:
            image (Image): The image to inspect.

        Returns:
            int: The extracted key if found, else None.
        """
        return self.extract_watermark(image)
    
    def embed_watermark(self, image: Image, key: int) -> Image:
        """
        Embeds a watermark into the image using the provided key.

        Args:
            image (Image): The image to watermark.
            key (int): The watermark key.

        Returns:
            Image: Watermarked image.
        """
        # Convert image to numpy array
        np_image = np.array(image).astype(np.uint8)

        # Simple example: Modify the least significant bit (LSB) of the blue channel
        # with bits from the key
        key_bits = np.unpackbits(np.array([key], dtype=np.uint32).view(np.uint8))

        # Ensure the image has enough pixels to embed the key
        height, width, _ = np_image.shape
        total_pixels = height * width
        if total_pixels < len(key_bits):
            raise ValueError("Image is too small to embed the key.")
        
        # Flatten the blue channel
        blue_channel = np_image[:, :, 2].flatten()

        # Embed key bits into the LSB of the blue channel
        blue_channel[:len(key_bits)] = (blue_channel[:len(key_bits)] & ~1) | key_bits

        # Reshape and update the blue channel
        np_image[:, :, 2] = blue_channel.reshape(height, width)

        # Convert back to PIL Image
        watermarked_image = Image.fromarray(np_image)

        return watermarked_image

    def extract_watermark(self, image: Image) -> int:
        """
        Extracts the watermark key from the image.

        Args:
            image (Image): The image to extract the key from.

        Returns:
            int: The extracted key if found, else None.
        """
        # Convert image to numpy array
        np_image = np.array(image).astype(np.uint8)

        # Flatten the blue channel
        blue_channel = np_image[:, :, 2].flatten()

        # Extract LSBs from the blue channel
        lsb_bits = blue_channel & 1

        # Only attempt to extract 32 bits (size of uint32)
        key_bits = lsb_bits[:32]

        if len(key_bits) < 32:
            return None  # Not enough data to extract key

        # Pack bits into bytes and convert to uint32
        key_bytes = np.packbits(key_bits)
        key = key_bytes.view(np.uint32)[0]

        return key
