"""Command-line training entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eat_bart.training.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the standard BART baseline.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(config_path=args.config)
