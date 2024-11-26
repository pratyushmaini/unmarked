# metrics/lpips.py

import torch
from lpips import LPIPS
from torchvision import transforms

def load_perceptual_models(device=torch.device("cuda")):
    perceptual_model = LPIPS(net="alex").to(device)
    return perceptual_model

def compute_perceptual_metric_repeated(model, images1, images2, device=torch.device("cuda")):
    scores = []
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    
    for img1, img2 in zip(images1, images2):
        # Convert PIL image to tensor
        img1_tensor = preprocess(img1).unsqueeze(0).to(device)
        img2_tensor = preprocess(img2).unsqueeze(0).to(device)

        # Compute LPIPS score
        score = model(img1_tensor, img2_tensor).item()
        scores.append(score)

    return sum(scores) / len(scores)
