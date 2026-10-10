"""Validate an uploaded baseline artifact before starting EAT training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eat_bart.training.comparability import (
    resolve_baseline_checkpoint,
    validate_baseline_manifest,
    validate_runtime,
)
from eat_bart.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/kaggle_encoder_eat_comparable.yaml",
        help="Resolved EAT comparison configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    validate_runtime(config)
    dataset_path = Path(config["data"]["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing comparison dataset: {dataset_path}")
    model_config = config["model"]
    checkpoint_path = resolve_baseline_checkpoint(
        model_config["baseline_checkpoint_path"],
        artifact_name=model_config["baseline_artifact_name"],
    )
    validate_baseline_manifest(checkpoint_path, config, dataset_path)
    print("Baseline/EAT comparability check passed.")
    print(f"Baseline checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
