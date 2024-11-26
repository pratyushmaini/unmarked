import os
import torch
import torch.nn as nn
from urllib.request import urlretrieve
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def get_aesthetic_model(clip_model="vit_l_14", device=torch.device("cuda")):
    """Load the aesthetic model for CLIP."""
    
    # Define model URL and path
    home = os.path.expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + f"/sa_0_4_{clip_model}_linear.pth"
    
    # Download the model weights if not already cached
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            f"https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_{clip_model}_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    
    # Initialize the linear layer based on the model type
    if clip_model == "vit_l_14":
        aesthetic_model = nn.Linear(768, 1)  # for ViT-Large
    elif clip_model == "vit_b_32":
        aesthetic_model = nn.Linear(512, 1)  # for ViT-Base
    else:
        raise ValueError(f"Unsupported clip_model: {clip_model}")
    
    # Load the weights
    state_dict = torch.load(path_to_model, map_location=device)
    aesthetic_model.load_state_dict(state_dict)
    aesthetic_model.to(device)  # Move to specified device (CPU or CUDA)
    aesthetic_model.eval()  # Set to evaluation mode
    
    return aesthetic_model


def compute_aesthetics_score(image: Image.Image, aesthetic_model, clip_model="vit_l_14", device=torch.device("cuda")):
    """Compute the aesthetic score for a given image using the pre-trained aesthetic model."""
    
    # Load the CLIP model and processor
    if clip_model == "vit_l_14":
        name = "openai/clip-vit-large-patch14"
    clip_processor = CLIPProcessor.from_pretrained(name)
    clip_model = CLIPModel.from_pretrained(name).to(device)

    # Preprocess the image for CLIP model
    inputs = clip_processor(images=image, return_tensors="pt").to(device)

    # Extract embeddings using CLIP model
    with torch.no_grad():
        clip_embeddings = clip_model.get_image_features(**inputs)

    # Normalize the embeddings (important for consistency)
    clip_embeddings = clip_embeddings / clip_embeddings.norm(p=2, dim=-1, keepdim=True)

    # Compute the aesthetic score using the pre-trained linear model
    with torch.no_grad():
        score = aesthetic_model(clip_embeddings)
    
    return score.item()  # Return the score as a scalar value
