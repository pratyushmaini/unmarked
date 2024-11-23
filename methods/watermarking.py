# methods/watermarking.py

from PIL import Image, ImageDraw, ImageFont

class SimpleWatermark:
    def __init__(self, text="Watermark", opacity=128, position=(0, 0)):
        self.text = text
        self.opacity = opacity
        self.position = position

    def apply(self, image: Image) -> Image:
        watermark = Image.new('RGBA', image.size)
        draw = ImageDraw.Draw(watermark)

        font = ImageFont.load_default()
        text_size = draw.textsize(self.text, font)
        position = self.position

        draw.text(position, self.text, fill=(255, 255, 255, self.opacity), font=font)
        combined = Image.alpha_composite(image.convert('RGBA'), watermark)
        return combined.convert('RGB')

class InvisibleWatermark:
    def __init__(self, key=12345):
        self.key = key  # Simple key for demonstration

    def apply(self, image: Image) -> Image:
        # Simple invisible watermark by modifying LSB of pixels
        np_image = np.array(image)
        # Embed key into the least significant bit of the first pixel
        np_image[0,0,0] = (np_image[0,0,0] & ~1) | (self.key & 1)
        return Image.fromarray(np_image)