from PIL import Image
from .base_attack import BaseAttack

class AdversarialPatchAttack(BaseAttack):
    def __init__(self, patch_size=50, position=(0,0)):
        super().__init__()
        self.patch_size = patch_size
        self.position = position  # (x, y) coordinates

    def apply(self, image: Image) -> Image:
        patched_image = image.copy()
        patch = Image.new('RGB', (self.patch_size, self.patch_size), color=(255, 0, 0))  # Red patch
        patched_image.paste(patch, self.position)
        return patched_image