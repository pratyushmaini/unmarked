# attacks/gaussian_noise_attack.py

import numpy as np
from PIL import Image
from .base_attack import BaseAttack

class GaussianNoiseAttack(BaseAttack):
    def __init__(self, mean=0, std=25):
        super().__init__()
        self.mean = mean
        self.std = std

    def apply(self, image: Image) -> Image:
        np_image = np.array(image).astype(np.float32)
        noise = np.random.normal(self.mean, self.std, np_image.shape)
        noised_image = np_image + noise
        noised_image = np.clip(noised_image, 0, 255).astype(np.uint8)
        return Image.fromarray(noised_image)