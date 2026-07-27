"""Command-line evaluation entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eat_bart.training.evaluate import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the standard BART baseline.")
    parser.add_argument(
        "--config",
        default="configs/evaluate.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(config_path=args.config)
