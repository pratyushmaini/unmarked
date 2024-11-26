import random
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import torch
import io
from .base_attack import BaseAttack

# Code adapted from https://github.com/umd-huang-lab/WAVES


# Normalize image tensors
def normalize_tensor(images, norm_type):
    assert norm_type in ["imagenet", "naive"]
    # Two possible normalization conventions
    if norm_type == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = T.Normalize(mean, std)
    elif norm_type == "naive":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = T.Normalize(mean, std)
    else:
        assert False
    return torch.stack([normalize(image) for image in images])


# Unnormalize image tensors
def unnormalize_tensor(images, norm_type):
    assert norm_type in ["imagenet", "naive"]
    # Two possible normalization conventions
    if norm_type == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        unnormalize = T.Normalize(
            (-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
            (1 / std[0], 1 / std[1], 1 / std[2]),
        )
    elif norm_type == "naive":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unnormalize = T.Normalize(
            (-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
            (1 / std[0], 1 / std[1], 1 / std[2]),
        )
    else:
        assert False
    return torch.stack([unnormalize(image) for image in images])


# Convert PIL images to tensors and normalize
def to_tensor(images, norm_type="naive"):
    assert isinstance(images, list) and all(
        [isinstance(image, Image.Image) for image in images]
    )
    images = torch.stack([T.ToTensor()(image) for image in images])
    if norm_type is not None:
        images = normalize_tensor(images, norm_type)
    return images


# Unnormalize tensors and convert to PIL images
def to_pil(images, norm_type="naive"):
    assert isinstance(images, torch.Tensor)
    if norm_type is not None:
        images = unnormalize_tensor(images, norm_type).clamp(0, 1)
    return [T.ToPILImage()(image) for image in images.cpu()]


# Renormalize image tensors
def renormalize_tensor(images, in_norm_type=None, out_norm_type=None):
    assert in_norm_type in ["imagenet", "naive"]
    assert out_norm_type in ["imagenet", "naive"]

    # First unnormalize the tensor using the input normalization type
    images = unnormalize_tensor(images, in_norm_type)

    # Then normalize the tensor using the output normalization type
    images = normalize_tensor(images, out_norm_type)

    return images


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


distortion_strength_paras = dict(
    rotation=(0, 45),
    resizedcrop=(1, 0.5),
    erasing=(0, 0.25),
    brightness=(1, 2),
    contrast=(1, 2),
    blurring=(0, 20),
    noise=(0, 0.1),
    compression=(90, 10),
)


def relative_strength_to_absolute(strength, distortion_type):
    assert 0 <= strength <= 1
    strength = (
        strength
        * (
            distortion_strength_paras[distortion_type][1]
            - distortion_strength_paras[distortion_type][0]
        )
        + distortion_strength_paras[distortion_type][0]
    )
    strength = max(strength, min(*distortion_strength_paras[distortion_type]))
    strength = min(strength, max(*distortion_strength_paras[distortion_type]))
    return strength


def apply_distortion(
    images,
    distortion_type,
    strength=None,
    distortion_seed=0,
    same_operation=False,
    relative_strength=True,
    return_image=True,
):
    # Convert images to PIL images if they are tensors
    if not isinstance(images[0], Image.Image):
        images = to_pil(images)
    # Check if strength is relative and convert if needed
    if relative_strength:
        strength = relative_strength_to_absolute(strength, distortion_type)
    # Apply distortions
    distorted_images = []
    seed = distortion_seed
    for image in images:
        distorted_images.append(
            apply_single_distortion(
                image, distortion_type, strength, distortion_seed=seed
            )
        )
        # If not applying the same distortion, increment the seed
        if not same_operation:
            seed += 1
    # Convert to tensors if needed
    if not return_image:
        distorted_images = to_tensor(distorted_images)
    return distorted_images


def apply_single_distortion(image, distortion_type, strength=None, distortion_seed=0):
    # Accept a single image
    assert isinstance(image, Image.Image)
    # Set the random seed for the distortion if given
    set_random_seed(distortion_seed)
    # Assert distortion type is valid
    assert distortion_type in distortion_strength_paras.keys()
    # Assert strength is in the correct range
    if strength is not None:
        assert (
            min(*distortion_strength_paras[distortion_type])
            <= strength
            <= max(*distortion_strength_paras[distortion_type])
        )

    # Apply the distortion
    if distortion_type == "rotation":
        angle = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["rotation"])
        )
        distorted_image = F.rotate(image, angle)

    elif distortion_type == "resizedcrop":
        scale = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["resizedcrop"])
        )
        i, j, h, w = T.RandomResizedCrop.get_params(
            image, scale=(scale, scale), ratio=(1, 1)
        )
        distorted_image = F.resized_crop(image, i, j, h, w, image.size)

    elif distortion_type == "erasing":
        scale = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["erasing"])
        )
        image = to_tensor([image], norm_type=None)
        i, j, h, w, v = T.RandomErasing.get_params(
            image, scale=(scale, scale), ratio=(1, 1), value=[0]
        )
        distorted_image = F.erase(image, i, j, h, w, v)
        distorted_image = to_pil(distorted_image, norm_type=None)[0]

    elif distortion_type == "brightness":
        factor = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["brightness"])
        )
        enhancer = ImageEnhance.Brightness(image)
        distorted_image = enhancer.enhance(factor)

    elif distortion_type == "contrast":
        factor = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["contrast"])
        )
        enhancer = ImageEnhance.Contrast(image)
        distorted_image = enhancer.enhance(factor)

    elif distortion_type == "blurring":
        kernel_size = (
            int(strength)
            if strength is not None
            else random.uniform(*distortion_strength_paras["blurring"])
        )
        distorted_image = image.filter(ImageFilter.GaussianBlur(kernel_size))

    elif distortion_type == "noise":
        std = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["noise"])
        )
        image = to_tensor([image], norm_type=None)
        noise = torch.randn(image.size()) * std
        distorted_image = to_pil((image + noise).clamp(0, 1), norm_type=None)[0]

    elif distortion_type == "compression":
        quality = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["compression"])
        )
        quality = int(quality)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=quality)
        distorted_image = Image.open(buffered)

    else:
        assert False

    return distorted_image


class DistortionAttack(BaseAttack):
    def __init__(self, distortion_type: str, strength: float = None):
        super().__init__()
        self.distortion_type = distortion_type
        self.strength = strength

    def apply(self, image: Image) -> Image:
        print(
            f"Applying distortion attack {self.distortion_type} with strength {self.strength}"
        )
        return apply_single_distortion(
            image,
            self.distortion_type,
            self.strength,
            distortion_seed=0,
        )


class ChainedDistortionAttack(BaseAttack):
    def __init__(self, distortion_types: list[str], strengths: list[float] = None):
        super().__init__()
        if strengths is None:
            strengths = [None] * len(distortion_types)
        self.attacks = [
            DistortionAttack(distortion_type, strength)
            for distortion_type, strength in zip(distortion_types, strengths)
        ]

    def apply(self, image: Image) -> Image:
        for attack in self.attacks:
            image = attack.apply(image)
        return image
