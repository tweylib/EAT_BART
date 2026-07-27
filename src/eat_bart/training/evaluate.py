"""Evaluation helpers for the standard BART baseline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from transformers import BartForConditionalGeneration

from eat_bart.data.dataset import MentalHealthResponseDataset, split_dataset
from eat_bart.data.tokenizer import load_bart_tokenizer
from eat_bart.training.train import DEFAULT_MODEL_NAME, _require_file
from eat_bart.utils.config import load_yaml_config
from eat_bart.utils.seed import set_seed


def evaluate(config_path: str | Path = "configs/evaluate.yaml") -> None:
    """Evaluate a fine-tuned or pretrained standard BART model."""
    config = load_yaml_config(config_path)
    run_generation(config)


def run_generation(config: dict[str, Any]) -> Path:
    """Generate responses from standard BART and write them to CSV."""
    model_config = config["model"]
    data_config = config["data"]
    evaluation_config = config["evaluation"]
    training_config = config.get("training", {})
    seed = int(training_config.get("seed", 42))
    set_seed(seed)

    dataset_path = _require_file(data_config["dataset_path"], "dataset CSV")
    model_source = evaluation_config.get("model_source", "checkpoint")
    checkpoint_path = None
    if model_source == "checkpoint":
        checkpoint_path = _require_file(evaluation_config["checkpoint_path"], "BART checkpoint")

    dataset = MentalHealthResponseDataset.from_csv(
        path=dataset_path,
        question_column=data_config.get("question_column", "question"),
        response_column=data_config.get("response_column", "response"),
        limit=data_config.get("max_examples"),
    )
    _, _, test_dataset = split_dataset(
        dataset,
        validation_size=float(data_config.get("validation_size", 0.1)),
        test_size=float(data_config.get("test_size", 0.1)),
        seed=seed,
    )
    examples = [test_dataset[index] for index in range(len(test_dataset))]
    max_eval_examples = evaluation_config.get("max_eval_examples")
    if max_eval_examples is not None:
        examples = examples[: int(max_eval_examples)]

    model_name = model_config.get("name", DEFAULT_MODEL_NAME)
    local_files_only = bool(model_config.get("local_files_only", False))
    tokenizer = load_bart_tokenizer(
        model_name,
        local_files_only=local_files_only,
        add_prefix_space=bool(model_config.get("add_prefix_space", False)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_source == "checkpoint":
        model = BartForConditionalGeneration.from_pretrained(
            checkpoint_path,
        ).to(device)
    elif model_source == "pretrained_bart":
        model = BartForConditionalGeneration.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(device)
    else:
        raise ValueError(f"Unsupported evaluation model_source: {model_source}")
    model.eval()

    rows = _generate_rows(
        examples=examples,
        tokenizer=tokenizer,
        model=model,
        device=device,
        data_config=data_config,
        evaluation_config=evaluation_config,
    )

    output_path = Path(evaluation_config.get("output_path", "reports/bart_baseline_generations.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_generation_rows(output_path, rows)
    return output_path


def _generate_rows(
    examples: list[dict[str, str]],
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    data_config: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    batch_size = int(evaluation_config.get("batch_size", 4))

    for start_index in range(0, len(examples), batch_size):
        batch_examples = examples[start_index : start_index + batch_size]
        questions = [example["question"] for example in batch_examples]
        references = [example["response"] for example in batch_examples]

        encoded = tokenizer(
            questions,
            max_length=int(data_config.get("max_source_length", 256)),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            generation_kwargs = _build_generation_kwargs(evaluation_config)
            generated_ids = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                **generation_kwargs,
            )

        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions_with_special_tokens = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )
        for batch_index, (question, reference, prediction) in enumerate(
            zip(questions, references, predictions, strict=True)
        ):
            token_ids = generated_ids[batch_index].detach().cpu().tolist()
            rows.append(
                {
                    "question": question,
                    "reference_response": reference,
                    "generated_response": prediction,
                    "generated_response_with_special_tokens": predictions_with_special_tokens[batch_index],
                    "generated_token_ids": " ".join(str(token_id) for token_id in token_ids),
                }
            )

    return rows


def _build_generation_kwargs(evaluation_config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(evaluation_config.get("max_new_tokens", 128)),
        "num_beams": int(evaluation_config.get("num_beams", 4)),
        "do_sample": bool(evaluation_config.get("do_sample", False)),
    }
    optional_keys = {
        "min_new_tokens": int,
        "repetition_penalty": float,
        "no_repeat_ngram_size": int,
        "length_penalty": float,
        "early_stopping": bool,
        "temperature": float,
        "top_p": float,
        "top_k": int,
    }
    for key, caster in optional_keys.items():
        value = evaluation_config.get(key)
        if value is not None:
            kwargs[key] = caster(value)

    return kwargs


def _write_generation_rows(output_path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "question",
        "reference_response",
        "generated_response",
        "generated_response_with_special_tokens",
        "generated_token_ids",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
