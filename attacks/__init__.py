# attacks/__init__.py

from .base_attack import BaseAttack, NoAttack
from .gaussian_noise_attack import GaussianNoiseAttack
from .adversarial_patch_attack import AdversarialPatchAttack

__all__ = [
    'BaseAttack',
    'GaussianNoiseAttack',
    'AdversarialPatchAttack',
    'NoAttack'
]