# Unmarked Challenge

A benchmark challenge focused on washing out image watermarks. Teams compete to either create robust watermarks that are hard to remove or develop methods to remove watermarks while maintaining image quality under certain constraints.

## Teams & Objectives

### Blue Team
- **Goal**: Embed watermarks into images that are difficult for the Red Team to remove without significantly degrading image quality.
- **Tasks**:
  - **Watermarking Methods**: Implement robust watermarking techniques in `methods/watermarking.py`.
  - **Detection Methods**: Develop detection algorithms in `methods/detection.py` to verify the presence of watermarks in images.

### Red Team
- **Goal**: Remove or obscure watermarks from images provided by the Blue Team while keeping image distortions within acceptable limits.
- **Tasks**:
  - **Attack Strategies**: Implement watermark removal attacks in `attacks/`.
  - **Constraints**: Must maintain high image quality and avoid excessive distortions that would be noticeable or detectable.

## Evaluation

- **Effectiveness of Watermarking**:
  - How well the Blue Team's watermarks resist removal attempts.
  - Detection rate of watermarks after Red Team's attacks.
- **Attack Success Rate**:
  - Ability of the Red Team to remove watermarks without causing significant image degradation.
- **Image Quality Metrics**:
  - **LPIPS**: Measures perceptual similarity between original and attacked images.
  - **Clean FID**: Evaluates distribution similarity between sets of images.
  - **Aesthetics Score**: Assesses visual appeal of the images.
- **Area Under the Curve (AUC)**:
  - Combines detection rates and image quality metrics to provide an overall performance score.

## Repository Structure

```
unmarked/
├── attacks/          # Red Team's attack implementations
├── data/             # Datasets of watermarked and clean images
├── methods/          # Blue Team's watermarking and detection methods
├── metrics/          # Image quality and performance metrics
├── scripts/          # Utility scripts for data processing and reporting
├── README.md         # Project documentation
├── battle.py         # Manages the competition between teams
├── main.py           # Entry point to run the benchmark
├── registry.py       # Registers available methods, attacks, and teams
├── requirements.txt  # Project dependencies
└── setup.sh          # Setup script for initializing the environment
```

## Getting Started

### 1. Clone and Setup the Repository

```bash
git clone https://github.com/yourusername/unmarked.git
cd unmarked
bash setup.sh
```

- **Note**: The `setup.sh` script creates a virtual environment and installs all required dependencies listed in `requirements.txt`.

### 2. Understand the Codebase

Run the following commands to get familiar with the basic functionalities:

#### Example 1: Running a Battle with No Defense and No Attack

```bash
python main.py --blue_team NoWatermarkTeam --red_team NoAttackTeam --image_path data/clean_images/sample.jpg
```

- **Expected Outcome**: The original image is processed without any watermarking or attack. No watermark should be detected.

#### Example 2: Blue Team Adds Watermark, Red Team Does Not Attack

```bash
python main.py --blue_team BaseBlueTeam --red_team NoAttackTeam --image_path data/clean_images/sample.jpg
```

- **Expected Outcome**: The image is watermarked by the Blue Team. The watermark should be detectable by the Blue Team's detection method.

#### Example 3: Blue Team Adds Watermark, Red Team Attempts to Remove It

```bash
python main.py --blue_team BaseBlueTeam --red_team BaseRedTeam --image_path data/clean_images/sample.jpg
```

- **Expected Outcome**: The Red Team applies a Gaussian noise attack to remove the watermark. Evaluate whether the watermark is still detectable and assess the image quality.

### 3. Implement Your Strategy

#### For Blue Team:

- **Watermarking Methods**: Enhance or create new methods in `methods/watermarking.py`.
- **Detection Methods**: Improve detection algorithms in `methods/detection.py`.
- **Register Your Team**: Add your team configuration to `registry.py` under `STUDENT_TEAMS`.

#### For Red Team:

- **Attack Strategies**: Implement new attacks in the `attacks/` directory.
- **Constraints**: Ensure that image distortions stay within acceptable limits.
- **Register Your Team**: Add your team configuration to `registry.py` under `STUDENT_TEAMS`.

#### Steps:

1. **Develop** your methods or attacks in the appropriate directories.
2. **Register** your team in `registry.py`:

   ```python
   STUDENT_TEAMS = {
       'YourTeamNameBlue': {
           'type': 'blue',
           'watermark_method': 'YourWatermarkMethod',
           'detection_method': 'YourDetectionMethod'
       },
       'YourTeamNameRed': {
           'type': 'red',
           'attack_method': 'YourAttackMethod'
       },
   }
   ```

3. **Test** your implementations by running battles using `main.py`.
4. **Document** your approach and findings in the README.

### 4. Evaluate and Iterate

- **Run Battles**: Use different combinations of Red and Blue teams to test the effectiveness of your methods.
- **Assess Performance**: Utilize the metrics in the `metrics/` directory to evaluate image quality and watermark detectability.
- **Optimize**: Based on the results, refine your methods to improve performance.

### 5. Submit Your Work

- **Fork** the repository.
- **Commit** your changes, ensuring that your code is well-documented and follows the project's style guidelines.
- **Document** your strategies and results in the README.
- **Create a Pull Request** to submit your team's implementation for review.

## Strategies to Consider

### Blue Team

- **Robust Watermarking**:
  - Use invisible watermarking techniques that are difficult to detect or remove.
  - Employ frequency-domain watermarking to embed watermarks in less noticeable parts of the image spectrum.
- **Advanced Detection**:
  - Implement machine learning models to detect subtle changes indicative of watermark removal.
  - Use statistical analysis to identify anomalies in images that have undergone attacks.

### Red Team

- **Image Processing Attacks**:
  - Apply filters, noise reduction, or transformations to remove or degrade the watermark.
- **Adversarial Attacks**:
  - Generate adversarial examples that fool the detection algorithms without significant image distortion.
- **Optimization-Based Methods**:
  - Use optimization algorithms to find minimal perturbations that remove the watermark.

