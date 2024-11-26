# methods/output_pixel_watermarking.py

from methods.watermarked_diffusion_pipeline import BaseWatermarkedDiffusionPipeline
from PIL import Image

class OutputPixelWatermarking(BaseWatermarkedDiffusionPipeline):
    def generate(self, prompt: str, key: int = None, **generate_kwargs):
        # Generate image using the base pipeline
        image = super().generate(prompt, key=None, **generate_kwargs)
        if key is not None:
            # Embed the key into a specific pixel
            image = self.embed_watermark(image, key)
        return image

    def embed_watermark(self, image: Image.Image, key: int) -> Image.Image:
        # Embed the key into the top-left pixel (0,0)
        pixels = image.load()
        x, y = 0, 0
        # Convert key to RGB tuple
        r = (key >> 16) & 0xFF
        g = (key >> 8) & 0xFF
        b = key & 0xFF
        pixels[x, y] = (r, g, b)
        return image

    def detect(self, image: Image.Image) -> int:
        # Extract the key from the specific pixel
        pixels = image.load()
        x, y = 0, 0
        r, g, b = pixels[x, y]
        key = (r << 16) | (g << 8) | b
        return key
