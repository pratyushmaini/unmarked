import torch
import argparse
from pathlib import Path
from battle import Battle
from calculate_metrics import calculate_metrics
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Unmarked Benchmark Runner")
    parser.add_argument('--red_team', type=str, required=True, help="Name of the Red Team")
    parser.add_argument('--blue_team', type=str, required=True, help="Name of the Blue Team")
    parser.add_argument('--prompt_file', type=str, required=True, help="File containing prompts for image generation")
    parser.add_argument('--output_dir', type=str, default='data', help="Directory to save outputs")
    parser.add_argument('--optimize_memory', action='store_true', help="Optimize model memory usage")
    # compute_metrics?
    parser.add_argument('--compute_metrics', action='store_true', help="Compute metrics for generated and attacked images")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Battle
    battle = Battle(output_dir=args.output_dir, optimize_memory=args.optimize_memory)

    # Run battles for all prompts in the file
    with open(args.prompt_file, 'r') as file:
        prompts = [line.strip() for line in file.readlines()]

    results = battle.run_battles(args.red_team, args.blue_team, prompts)

    # Paths for generated and attacked images
    generated_images_dir = Path(args.output_dir) / "generated_images"
    attacked_images_dir = Path(args.output_dir) / "attacked_images"

    # Calculate metrics
    if args.compute_metrics:
        metrics = calculate_metrics(generated_images_dir, attacked_images_dir, device)
        # Report metrics
        print("\n=== Metrics ===")
        print(f"Average Generated Image Aesthetic Score: {metrics['aesthetic_generated']}")
        print(f"Average Attacked Image Aesthetic Score: {metrics['aesthetic_attacked']}")
        print(f"Average LPIPS Score: {metrics['lpips_score']}")
        print(f"FID Score (Generated vs Reference): {metrics['fid_generated']}")
        print(f"FID Score (Attacked vs Reference): {metrics['fid_attacked']}")


if __name__ == "__main__":
    main()
