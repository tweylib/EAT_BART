"""Command-line scoring entry point for generated responses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eat_bart.training.metrics import score_generation_csv
from eat_bart.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score generated response CSV.")
    parser.add_argument(
        "--config",
        default="configs/kaggle_encoder_only_score.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    scoring_config = config["scoring"]
    metrics = score_generation_csv(
        input_path=scoring_config["input_path"],
        output_path=scoring_config.get("output_path"),
        prediction_column=scoring_config.get("prediction_column", "generated_response"),
        reference_column=scoring_config.get("reference_column", "reference_response"),
        include_bertscore=bool(scoring_config.get("include_bertscore", False)),
        bertscore_model_type=scoring_config.get(
            "bertscore_model_type",
            "distilbert-base-uncased",
        ),
        bertscore_batch_size=int(scoring_config.get("bertscore_batch_size", 16)),
        validation_loss_path=scoring_config.get("validation_loss_path"),
        validation_loss_mode=scoring_config.get("validation_loss_mode", "best"),
    )

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
