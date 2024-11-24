# Unmarked Challenge

**Unmarked** is a benchmark challenge focused on washing out image watermarks embedded within the outputs of the Stable Diffusion XL (SDXL) model. Participants will engage in a Red Team vs. Blue Team scenario, where the Blue Team embeds watermarks into the generated images, and the Red Team attempts to remove these watermarks under certain constraints.

## Objective

- **Blue Team (Defenders)**: Implement robust watermarking techniques within the diffusion model that are difficult for the Red Team to remove without significantly degrading image quality.
- **Red Team (Attackers)**: Develop methods to remove or obscure the embedded watermarks while maintaining high image quality and adhering to distortion constraints.

## Repository Structure

```
unmarked/
├── attacks/                # Red Team's attack implementations
│   ├── __init__.py
│   └── base_attack.py
├── data/                   # Datasets for testing
├── methods/                # Blue Team's watermarking and detection methods
│   ├── __init__.py
│   └── watermarked_pipeline.py
├── metrics/                # Evaluation metrics
│   ├── __init__.py
│   ├── lpips_metric.py
│   ├── aesthetics.py
│   └── auc_metric.py
├── scripts/                # Utility scripts
├── registry.py             # Manages available methods, attacks, and teams
├── battle.py               # Manages the competition between teams
├── main.py                 # Entry point to run the benchmark
├── README.md               # Project documentation
├── requirements.txt        # Project dependencies
└── setup.sh                # Setup script for initializing the environment
```

## Key Components

- **`registry.py`**: Central registry managing all available methods, attacks, and team configurations.
- **`battle.py`**: Orchestrates the battle between the Red and Blue teams using configurations from the registry.
- **`methods/`**: Contains the `WatermarkedPipeline` class that extends the SDXL pipeline for watermarking and detection.
- **`attacks/`**: Contains attack implementations by the Red Team.
- No changes are to be made to main files (`battle.py`, `main.py`, `metrics`), ensuring a standardized interface for all participants.

## Getting Started

### 1. Clone and Setup the Repository

```bash
git clone https://github.com/yourusername/unmarked.git
cd unmarked
conda create -n unmarked python=3.11 -y
pip install -r requirements.txt
```


### 2. Understand the Codebase Quickly

#### Key Files and Their Roles

- **`registry.py`**: Manages all available methods, attacks, and team configurations.
- **`battle.py`**: Manages the execution flow of generating images, applying attacks, and detecting watermarks.
- **`main.py`**: Entry point for running the benchmark and testing your implementations.

#### Quick Code Exploration Commands

1. **Run a Simple Battle with Baseline Teams**

   ```bash
   python main.py --red_team NoAttackTeam --blue_team NoWatermarkTeam --prompt "A serene mountain landscape at sunrise"
   ```

   - **Expected Outcome**: Generates an image without any watermarking or attack. No watermark should be detected.

2. **Blue Team Adds Watermark, Red Team Does Not Attack**

   ```bash
   python main.py --red_team NoAttackTeam --blue_team BaseBlueTeam --prompt "A serene mountain landscape at sunrise"
   ```

   - **Expected Outcome**: The image is watermarked by the Blue Team. The watermark should be detectable by the Blue Team's detection method.

3. **Blue Team Adds Watermark, Red Team Attempts to Remove It**

   ```bash
   python main.py --red_team BaseRedTeam --blue_team BaseBlueTeam --prompt "A serene mountain landscape at sunrise"
   ```

   - **Expected Outcome**: The Red Team applies their attack to remove the watermark. Evaluate whether the watermark is still detectable and assess the image quality.


### 3. Implement Your Strategy

#### For Blue Team (Defenders)

- **Objective**: Implement robust watermarking and detection methods without modifying main files.
- **Steps**:

  1. **Create Your Watermarking Method**

     - In `methods/`, create a new file for your watermarking method, e.g., `my_watermarking.py`.

     ```python
     # methods/my_watermarking.py

     from methods.watermarked_diffusion_pipeline import BaseWatermarkedDiffusionPipeline

     class MyWatermarkedPipeline(BaseWatermarkedDiffusionPipeline):
         def __init__(self, *args, **kwargs):
             super().__init__(*args, **kwargs)

         def generate(self, prompt, key):
             # Your watermark embedding logic here
             return latents

         def detect(self, image):
             # Your watermark detection logic here
             return key
     ```

  

#### For Red Team (Attackers)

- **Objective**: Develop attack methods to remove or obscure watermarks.
- **Steps**:

  1. **Create Your Attack Method**

     - In `attacks/`, create a new file for your attack, e.g., `my_attack.py`.

     ```python
     # attacks/my_attack.py

     from attacks.base_attack import BaseAttack
     from PIL import Image

     class MyAttack(BaseAttack):
         def apply(self, image: Image) -> Image:
             # Your attack logic here
             return image
     ```

  2. **Register Your Attack in `registry.py`**

     ```python
     # registry.py

     BASELINE_METHODS = {
         'attacks': {
             'BaseAttack': 'attacks.base_attack.BaseAttack',
             'MyAttack': 'attacks.my_attack.MyAttack',
         },
         # ... other methods
     }

     # Register your team
     STUDENT_TEAMS = {
         'MyRedTeam': {
             'type': 'red',
             'attack_method': 'MyAttack',
         },
         # ... other teams
     }
     ```

### 3. Run Your Battle

```bash
python main.py --red_team MyRedTeam --blue_team MyBlueTeam --prompt "A futuristic cityscape at night"
```

- **Expected Outcome**: The battle will use your custom watermarking and attack methods as registered in `registry.py`.

### 4. Evaluate and Iterate

- **Metrics**:

  - **Watermark Detection Rate**: Whether the watermark is still detectable after the attack.
  - **Image Quality**: Use metrics like LPIPS to assess the perceptual similarity between the original and attacked images.
  - **Distortion Constraints**: Ensure your attacks do not introduce excessive distortions.

- **Run Metrics Evaluation**:

  ```bash
  # Example command to evaluate LPIPS
  python metrics/lpips_metric.py --original outputs/BaseBlueTeam_watermarked.png --modified outputs/MyRedTeam_vs_MyBlueTeam.png
  ```

### 5. Submit Your Work

- **Fork** the repository.
- **Commit** your changes, ensuring that you only add new files or modify `registry.py` and your own method/attack files.
- **Document** your strategies and findings in the README or a separate documentation file.
- **Create a Pull Request** to submit your implementation for review.
