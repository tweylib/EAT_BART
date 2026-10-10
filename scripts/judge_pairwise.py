"""Command-line entry point for blinded pairwise generation judging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eat_bart.training.pairwise_judge import judge_pairwise_generation_csv
from eat_bart.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge two aligned generation CSVs pairwise.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)["pairwise_judge"]
    summary = judge_pairwise_generation_csv(
        left_input_path=config["left_input_path"],
        right_input_path=config["right_input_path"],
        output_path=config["output_path"],
        summary_output_path=config.get("summary_output_path"),
        left_name=config.get("left_name", "left"),
        right_name=config.get("right_name", "right"),
        provider=config.get("provider", "groq"),
        model=config.get("model", "openai/gpt-oss-120b"),
        api_key_env=config.get("api_key_env"),
        question_column=config.get("question_column", "question"),
        prediction_column=config.get("prediction_column", "generated_response"),
        reference_column=config.get("reference_column", "reference_response"),
        max_examples=config.get("max_examples"),
        order_seed=int(config.get("order_seed", 20261010)),
        sleep_seconds=float(config.get("sleep_seconds", 0.0)),
        temperature=float(config.get("temperature", 0.0)),
        timeout_seconds=int(config.get("timeout_seconds", 60)),
        max_retries=int(config.get("max_retries", 3)),
        rate_limit_sleep_seconds=float(config.get("rate_limit_sleep_seconds", 65.0)),
        continue_on_error=bool(config.get("continue_on_error", False)),
        max_output_tokens=config.get("max_output_tokens"),
        response_format_json=bool(config.get("response_format_json", False)),
        reasoning_effort=config.get("reasoning_effort"),
    )
    for name, value in summary.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()
