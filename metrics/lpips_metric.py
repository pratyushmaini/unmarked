# metrics/lpips_metric.py

import lpips
from PIL import Image
import torch

class LPIPSMetric:
    def __init__(self, model='alex'):
        self.loss_fn = lpips.LPIPS(net=model)

    def evaluate(self, img1: Image, img2: Image) -> float:
        # Convert images to tensors
        img1_tensor = self._preprocess(img1)
        img2_tensor = self._preprocess(img2)
        with torch.no_grad():
            distance = self.loss_fn(img1_tensor, img2_tensor).item()
        return distance

    def _preprocess(self, image: Image) -> torch.Tensor:
        # Convert PIL Image to Tensor
        image = image.convert('RGB')
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)
        return tensor
