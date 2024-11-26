# calculate_metrics.py

from pathlib import Path
from PIL import Image
import torch
from metrics.aesthetics import get_aesthetic_model, compute_aesthetics_score
from metrics.lpips import load_perceptual_models, compute_perceptual_metric_repeated
from metrics.compute_fid import compute_fid


def calculate_metrics(generated_images_dir, attacked_images_dir, device):
    """Calculate metrics for generated and attacked images."""

    # Ensure reference images directory exists
    reference_images_dir = Path("data/reference_images")
    if not reference_images_dir.exists():
        raise FileNotFoundError("Reference images directory does not exist. Please generate the reference images first.")

    # Load models
    aesthetic_model = get_aesthetic_model(clip_model="vit_l_14", device=device)
    lpips_model = load_perceptual_models(device=device)

    # Load images
    generated_images = [Image.open(p).convert('RGB') for p in sorted(generated_images_dir.glob('*.png'))]
    attacked_images = [Image.open(p).convert('RGB') for p in sorted(attacked_images_dir.glob('*.png'))]

    # Compute metrics
    metrics = {}

    # Aesthetic scores
    metrics["aesthetic_generated"] = sum(
        compute_aesthetics_score(img, aesthetic_model, device=device) for img in generated_images
    ) / len(generated_images)

    metrics["aesthetic_attacked"] = sum(
        compute_aesthetics_score(img, aesthetic_model, device=device) for img in attacked_images
    ) / len(attacked_images)

    # LPIPS scores (average over all image pairs)
    lpips_scores = compute_perceptual_metric_repeated(lpips_model, generated_images, attacked_images, device=device)
    metrics["lpips_score"] = lpips_scores

    # Compute FID scores using pytorch-fid
    device_str = device.type if isinstance(device, torch.device) else device
    metrics["fid_generated"] = compute_fid(reference_images_dir, generated_images_dir, device=device_str)
    metrics["fid_attacked"] = compute_fid(reference_images_dir, attacked_images_dir, device=device_str)

    return metrics
