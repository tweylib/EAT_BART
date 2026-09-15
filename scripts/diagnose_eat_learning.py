"""Inspect EAT checkpoint learning, attention structure, and generation changes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eat_bart.training.eat_learning_diagnostic import diagnose_eat_learning
from eat_bart.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose encoder EAT learning dynamics.")
    parser.add_argument(
        "--config",
        default="configs/kaggle_encoder_eat_comparable_a005_lr1e4_diagnostic.yaml",
        help="Path to the diagnostic YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnose_eat_learning(load_yaml_config(args.config))


if __name__ == "__main__":
    main()
