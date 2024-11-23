# attacks/base_attack.py

from PIL import Image

class BaseAttack:
    def __init__(self):
        pass

    def apply(self, image: Image) -> Image:
        """
        Apply attack to the input image.
        Should be overridden by subclasses.
        """
        raise NotImplementedError("Attack method not implemented.")