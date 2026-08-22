"""CLI for generating synthetic training and evaluation datasets."""

import os
import argparse
from src.simulation.generator import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic recovery datasets.")
    parser.add_argument("--train-size", type=int, default=2000, help="Number of training rows (default: 2000)")
    parser.add_argument("--eval-size", type=int, default=500, help="Number of eval rows (default: 500)")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for CSV files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating training set ({args.train_size} rows)...")
    _, train_df = generate_dataset(
        num_events=args.train_size,
        seed=args.seed,
        sample_actions_per_event=True,
    )
    train_path = os.path.join(output_dir, "train.csv")
    train_df.to_csv(train_path, index=False)
    print(f"Saved training dataset to {train_path}")

    print(f"Generating held-out evaluation set ({args.eval_size} rows)...")
    _, eval_df = generate_dataset(
        num_events=args.eval_size,
        seed=args.seed + 100,
        sample_actions_per_event=True,
    )
    eval_path = os.path.join(output_dir, "eval.csv")
    eval_df.to_csv(eval_path, index=False)
    print(f"Saved evaluation dataset to {eval_path}")


if __name__ == "__main__":
    main()
