# battle.py

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
        self.output_dir.mkdir(exist_ok=True)
        self.optimize_memory = optimize_memory

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def run_battle(
        self, red_team_name: str, blue_team_name: str, prompt: str, key: int = 99
    ):
        """Execute a battle between teams"""

        # Get team configs
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

        # Initialize the pipeline
        watermarked_pipeline = WatermarkedPipelineClass()

        # Generate image with watermark
        generated_image = watermarked_pipeline.generate(prompt, key=key)
        generated_image_path = self.output_dir / f"{blue_team_name}_image.png"
        generated_image.save(generated_image_path)
        self.logger.info(f"Generated image saved to {generated_image_path}")

        # Apply Red Team's attack
        attack_name = red_config["attack_method"]
        attack_path = BASELINE_METHODS["attacks"][attack_name]
        module_path, class_name = attack_path.rsplit(".", 1)
        module = import_module(module_path)
        AttackClass = getattr(module, class_name)
        kwargs = {
            k: v
            for k, v in red_config.items()
            if (k != "type" and k != "attack_method")
        }
        attack = AttackClass(**kwargs)
        attacked_image = attack.apply(generated_image)
        attacked_image_path = (
            self.output_dir / f"{red_team_name}_vs_{blue_team_name}.png"
        )
        attacked_image.save(attacked_image_path)
        self.logger.info(f"Attacked image saved to {attacked_image_path}")

        # Detect watermark using the Blue Team's pipeline
        extracted_key = watermarked_pipeline.detect(attacked_image)
        detection_success = extracted_key == key
        self.logger.info(f"Extracted key: {extracted_key}")
        self.logger.info(f"Watermark detection successful: {detection_success}")

        return {
            "generated_image": generated_image,
            "attacked_image": attacked_image,
            "extracted_key": extracted_key,
            "detection_success": detection_success,
        }
