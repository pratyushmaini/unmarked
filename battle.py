import torch
from pathlib import Path
import logging
from PIL import Image
from importlib import import_module
from registry import BASELINE_METHODS, BASELINE_TEAMS, STUDENT_TEAMS

class BlueTeam:
    """Defender team implementing watermarking and detection"""

    def __init__(self, team_config):
        self.config = team_config
        self.watermark_method = self._load_method('watermark_method', 'watermarking')
        self.detection_method = self._load_method('detection_method', 'detection')

    def _load_method(self, method_type, method_category):
        """Dynamically load method class from config"""
        if method_type not in self.config or self.config[method_type] is None:
            return None

        method_name = self.config[method_type]
        method_path = BASELINE_METHODS[method_category][method_name]
        module_path, class_name = method_path.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)()

    def add_watermark(self, image: Image) -> Image:
        """Apply watermark to the image"""
        if self.watermark_method:
            return self.watermark_method.apply(image)
        return image

    def detect_watermark(self, image: Image) -> bool:
        """Detect watermark in the image"""
        if self.detection_method:
            return self.detection_method.detect(image)
        return False

class RedTeam:
    """Attacker team implementing watermark removal"""

    def __init__(self, team_config):
        self.config = team_config
        self.attack_method = self._load_attack()

    def _load_attack(self):
        """Dynamically load attack method"""
        if 'attack_method' not in self.config or self.config['attack_method'] is None:
            return None

        attack_name = self.config['attack_method']
        attack_path = BASELINE_METHODS['attacks'][attack_name]
        module_path, class_name = attack_path.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)()

    def remove_watermark(self, image: Image) -> Image:
        """Attempt to remove watermark from the image"""
        if self.attack_method:
            return self.attack_method.apply(image)
        return image

class Battle:
    """Manages battles between Red and Blue teams"""

    def __init__(self, output_dir='outputs'):
        self.logger = self._setup_logging()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def run_battle(self, blue_team_name: str, red_team_name: str, image: Image):
        """Execute a battle between teams"""

        # Get team configs
        blue_config = {**BASELINE_TEAMS, **STUDENT_TEAMS}.get(blue_team_name)
        red_config = {**BASELINE_TEAMS, **STUDENT_TEAMS}.get(red_team_name)

        if not blue_config or blue_config['type'] != 'blue':
            raise ValueError("Blue team not found or invalid")
        if not red_config or red_config['type'] != 'red':
            raise ValueError("Red team not found or invalid")

        # Initialize teams
        blue_team = BlueTeam(blue_config)
        red_team = RedTeam(red_config)

        # Battle sequence
        self.logger.info(f"Battle: {blue_team_name} vs {red_team_name}")

        # Blue team adds watermark
        watermarked_image = blue_team.add_watermark(image)
        watermarked_image_path = self.output_dir / f"{blue_team_name}_watermarked.png"
        watermarked_image.save(watermarked_image_path)
        self.logger.info(f"Watermarked image saved to {watermarked_image_path}")

        # Red team attempts to remove watermark
        attacked_image = red_team.remove_watermark(watermarked_image)
        attacked_image_path = self.output_dir / f"{red_team_name}_attacked.png"
        attacked_image.save(attacked_image_path)
        self.logger.info(f"Attacked image saved to {attacked_image_path}")

        # Blue team detects watermark
        detection_result = blue_team.detect_watermark(attacked_image)
        self.logger.info(f"Watermark detected after attack: {detection_result}")

        return {
            'watermarked_image': watermarked_image,
            'attacked_image': attacked_image,
            'detection_result': detection_result
        }