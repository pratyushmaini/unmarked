# main.py

import argparse
from battle import Battle

def parse_args():
    parser = argparse.ArgumentParser(description="Unmarked Benchmark Runner")
    parser.add_argument('--red_team', type=str, required=True, help="Name of the Red Team")
    parser.add_argument('--blue_team', type=str, required=True, help="Name of the Blue Team")
    parser.add_argument('--prompt', type=str, required=True, help="Prompt for image generation")
    parser.add_argument('--output_dir', type=str, default='outputs', help="Directory to save outputs")
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize Battle
    battle = Battle(output_dir=args.output_dir)

    # Run battle
    results = battle.run_battle(args.red_team, args.blue_team, args.prompt)

if __name__ == "__main__":
    main()
