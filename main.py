# main.py

import argparse
from battle import Battle

def parse_args():
    parser = argparse.ArgumentParser(description="Unmarked Benchmark Runner")
    parser.add_argument('--prompt', type=str, required=True, help="Prompt for image generation")
    parser.add_argument('--key', type=int, default=None, help="Integer key for watermarking")
    parser.add_argument('--output_dir', type=str, default='outputs', help="Directory to save outputs")
    parser.add_argument('--optimize_memory', action='store_true', help="Optimize model memory usage")
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize Battle
    battle = Battle(output_dir=args.output_dir, optimize_memory=args.optimize_memory)

    # Run battle
    results = battle.run_battle(prompt=args.prompt, key=args.key)

    # Additional metric evaluations can be added here

if __name__ == "__main__":
    main()
