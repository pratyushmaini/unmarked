# methods/my_watermarking.py

from watermarked_diffusion_pipeline import BaseWatermarkedDiffusionPipeline

class MyWatermarkedPipeline(BaseWatermarkedDiffusionPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def generate(self, prompt, key):
    #     # Your watermark embedding logic here
    #     return latents

    # def detect(self, image):
    #     # Your watermark detection logic here
    #     return key
    
    def generate_image(self, prompt, **generate_kwargs) -> None:
        image = self.model(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0,
            **generate_kwargs
        ).images[0]

        image.save("methods/prewatermark_output/intermediate.tiff")

if __name__ == "__main__":
    pipe = MyWatermarkedPipeline()
    pipe.generate_image("a pokemon")