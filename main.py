import argparse
from pathlib import Path
from PIL import Image
from battle import Battle

def parse_args():
    parser = argparse.ArgumentParser(description="Unmarked Benchmark Runner")
    parser.add_argument('--blue_team', type=str, required=True, help="Name of the Blue Team")
    parser.add_argument('--red_team', type=str, required=True, help="Name of the Red Team")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    parser.add_argument('--output_dir', type=str, default='outputs', help="Directory to save outputs")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load input image
    image = Image.open(args.image_path).convert('RGB')

    # Initialize Battle
    battle = Battle(output_dir=args.output_dir)

    # Run battle
    results = battle.run_battle(args.blue_team, args.red_team, image)

    # Further processing or metric evaluations can be added here

if __name__ == "__main__":
    main()