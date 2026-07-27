"""Command-line LLM judge entry point for generated responses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eat_bart.training.llm_judge import judge_generation_csv
from eat_bart.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge generated responses with an LLM.")
    parser.add_argument(
        "--config",
        default="configs/kaggle_encoder_only_5epoch_experiment_judge_groq.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    judge_config = config["llm_judge"]
    summary = judge_generation_csv(
        input_path=judge_config["input_path"],
        output_path=judge_config["output_path"],
        summary_output_path=judge_config.get("summary_output_path"),
        provider=judge_config.get("provider", "groq"),
        model=judge_config.get("model", "llama-3.3-70b-versatile"),
        api_key_env=judge_config.get("api_key_env"),
        question_column=judge_config.get("question_column", "question"),
        prediction_column=judge_config.get("prediction_column", "generated_response"),
        reference_column=judge_config.get("reference_column", "reference_response"),
        max_examples=judge_config.get("max_examples"),
        sleep_seconds=float(judge_config.get("sleep_seconds", 0.0)),
        temperature=float(judge_config.get("temperature", 0.0)),
        timeout_seconds=int(judge_config.get("timeout_seconds", 60)),
        max_retries=int(judge_config.get("max_retries", 3)),
        rate_limit_sleep_seconds=float(judge_config.get("rate_limit_sleep_seconds", 65.0)),
        continue_on_error=bool(judge_config.get("continue_on_error", False)),
        max_output_tokens=judge_config.get("max_output_tokens"),
        response_format_json=bool(judge_config.get("response_format_json", False)),
    )

    for name, value in summary.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
