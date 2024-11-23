# metrics/clean_fid.py


from pytorch_fid import fid_score
from pathlib import Path
import torch

class CleanFID:
    def __init__(self, real_images_dir: str):
        self.real_images_dir = Path(real_images_dir)

    def evaluate(self, generated_images_dir: str) -> float:
        generated_images_dir = Path(generated_images_dir)
        fid_value = fid_score.calculate_fid_given_paths(
            [str(self.real_images_dir), str(generated_images_dir)],
            batch_size=50,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dims=2048
        )
        return fid_value