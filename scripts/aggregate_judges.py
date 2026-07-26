"""Command-line entry point for aggregating LLM judge summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eat_bart.training.judge_aggregation import aggregate_judge_summaries
from eat_bart.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multiple LLM judge summaries.")
    parser.add_argument(
        "--config",
        default="configs/kaggle_encoder_only_judge_groq_aggregate.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    aggregate_config = config["judge_aggregation"]
    aggregate = aggregate_judge_summaries(
        judges=aggregate_config["judges"],
        output_path=aggregate_config["output_path"],
    )

    for name, value in aggregate.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()
