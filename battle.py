import torch
from pathlib import Path
import logging
from PIL import Image
from importlib import import_module
from registry import BASELINE_METHODS, BASELINE_TEAMS, STUDENT_TEAMS


class Battle:
    """Manages battles between Red and Blue teams"""

    def __init__(self, output_dir="outputs", optimize_memory=False):
        self.logger = self._setup_logging()
        self.output_dir = Path(output_dir)
        self.generated_images_dir = Path("data/generated_images")
        self.attacked_images_dir = Path("data/attacked_images")
        self.generated_images_dir.mkdir(parents=True, exist_ok=True)
        self.attacked_images_dir.mkdir(parents=True, exist_ok=True)
        self.optimize_memory = optimize_memory

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def run_battles(self, red_team_name: str, blue_team_name: str, prompts: list, key: int = 99):
        """
        Execute battles for a list of prompts and save all images.

        Args:
            red_team_name (str): Name of the Red Team.
            blue_team_name (str): Name of the Blue Team.
            prompts (list): List of prompts for image generation.
            key (int): Key for watermarking.

        Returns:
            List of results for each prompt.
        """
        results = []

        # Load team configurations
        blue_config = {**BASELINE_TEAMS, **STUDENT_TEAMS}.get(blue_team_name)
        red_config = {**BASELINE_TEAMS, **STUDENT_TEAMS}.get(red_team_name)

        if not blue_config or blue_config["type"] != "blue":
            raise ValueError("Blue team not found or invalid")
        if not red_config or red_config["type"] != "red":
            raise ValueError("Red team not found or invalid")

        # Initialize Blue Team's Watermarked Diffusion Pipeline
        wm_method_name = blue_config["watermark_method"]
        wm_method_path = BASELINE_METHODS["watermarking"][wm_method_name]
        module_path, class_name = wm_method_path.rsplit(".", 1)
        module = import_module(module_path)
        WatermarkedPipelineClass = getattr(module, class_name)

        watermarked_pipeline = WatermarkedPipelineClass()

        # Initialize Red Team's Attack
        attack_name = red_config['attack_method']
        attack_path = BASELINE_METHODS['attacks'][attack_name]
        module_path, class_name = attack_path.rsplit('.', 1)
        module = import_module(module_path)
        AttackClass = getattr(module, class_name)
        kwargs = {
            k: v
            for k, v in red_config.items()
            if (k != "type" and k != "attack_method")
        }
        attack = AttackClass(**kwargs)

        for idx, prompt in enumerate(prompts):
            # Generate image with watermark
            generated_image = watermarked_pipeline.generate(prompt, key=key)
            generated_image_path = self.generated_images_dir / f"generated_image_{idx}.png"
            generated_image.save(generated_image_path)
            self.logger.info(f"Generated image saved to {generated_image_path}")

            # Apply Red Team's attack
            attacked_image = attack.apply(generated_image)
            attacked_image_path = (
            self.attacked_images_dir / f"attacked_image_{idx}.png"
        )
            attacked_image.save(attacked_image_path)
            self.logger.info(f"Attacked image saved to {attacked_image_path}")

            # Detect watermark using the Blue Team's pipeline
            extracted_key = watermarked_pipeline.detect(attacked_image)
            detection_success = (extracted_key == key)
            self.logger.info(f"Extracted key: {extracted_key}")
            self.logger.info(f"Watermark detection successful: {detection_success}")

            results.append({
                'prompt': prompt,
                'generated_image': generated_image_path,
                'attacked_image': attacked_image_path,
                'extracted_key': extracted_key,
                'detection_success': detection_success
            })

        return results
