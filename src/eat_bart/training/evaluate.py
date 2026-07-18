"""Evaluation helpers for EAT-BART."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from eat_bart.data.dataset import MentalHealthResponseDataset, split_dataset
from eat_bart.data.emotion_features import tokenize_texts_with_emotion_features
from eat_bart.data.emotion_lexicon import load_nrc_lexicon
from eat_bart.modeling.eat_attention import EATAttentionConfig
from eat_bart.modeling.eat_bart_model import DEFAULT_MODEL_NAME, load_eat_bart_checkpoint
from eat_bart.training.train import _require_file
from eat_bart.utils.config import load_yaml_config
from eat_bart.utils.seed import set_seed


def evaluate(config_path: str | Path = "configs/evaluate.yaml") -> None:
    """Evaluate EAT-BART."""
    config = load_yaml_config(config_path)
    run_generation(config)


def run_generation(config: dict[str, Any]) -> Path:
    """Generate responses from a trained EAT-BART checkpoint and write them to CSV."""
    model_config = config["model"]
    data_config = config["data"]
    evaluation_config = config["evaluation"]
    training_config = config.get("training", {})
    seed = int(training_config.get("seed", 42))
    set_seed(seed)

    dataset_path = _require_file(data_config["dataset_path"], "dataset CSV")
    lexicon_path = _require_file(data_config["nrc_lexicon_path"], "NRC lexicon CSV")
    checkpoint_path = _require_file(evaluation_config["checkpoint_path"], "EAT-BART checkpoint")

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
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if not tokenizer.is_fast:
        raise ValueError("EAT-BART emotion alignment requires a fast Hugging Face tokenizer.")

    lexicon = load_nrc_lexicon(lexicon_path)
    eat_config = EATAttentionConfig(
        num_heads=12,
        emotion_dim=int(model_config.get("emotion_dim", 8)),
        emotion_hidden_dim=int(model_config.get("emotion_hidden_dim", 32)),
        alpha_init=float(model_config.get("alpha_init", 0.05)),
        formula=model_config.get("attention_formula", "additive"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_eat_bart_checkpoint(checkpoint_path, eat_config=eat_config).to(device)
    model.eval()

    rows = _generate_rows(
        examples=examples,
        tokenizer=tokenizer,
        lexicon=lexicon,
        model=model,
        device=device,
        data_config=data_config,
        evaluation_config=evaluation_config,
    )

    output_path = Path(evaluation_config.get("output_path", "reports/eat_bart_generations.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_generation_rows(output_path, rows)
    return output_path


def _generate_rows(
    examples: list[dict[str, str]],
    tokenizer: Any,
    lexicon: Any,
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

        encoded, encoder_emotion_features = tokenize_texts_with_emotion_features(
            texts=questions,
            tokenizer=tokenizer,
            lexicon=lexicon,
            max_length=int(data_config.get("max_source_length", 256)),
            padding=True,
            truncation=True,
            strategy=data_config.get("subword_strategy", "single"),
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        encoder_emotion_features = encoder_emotion_features.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                encoder_emotion_features=encoder_emotion_features,
                max_new_tokens=int(evaluation_config.get("max_new_tokens", 128)),
                num_beams=int(evaluation_config.get("num_beams", 4)),
                do_sample=bool(evaluation_config.get("do_sample", False)),
            )

        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for question, reference, prediction in zip(questions, references, predictions, strict=True):
            rows.append(
                {
                    "question": question,
                    "reference_response": reference,
                    "generated_response": prediction,
                }
            )

    return rows


def _write_generation_rows(output_path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["question", "reference_response", "generated_response"]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
