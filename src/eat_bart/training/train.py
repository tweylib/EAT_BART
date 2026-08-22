"""Training loop setup for the standard BART baseline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from transformers import (
    BartForConditionalGeneration,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from eat_bart.data.collator import BartDataCollator
from eat_bart.data.dataset import MentalHealthResponseDataset, split_dataset
from eat_bart.data.tokenizer import load_bart_tokenizer
from eat_bart.utils.config import load_yaml_config
from eat_bart.utils.seed import set_seed

DEFAULT_MODEL_NAME = "facebook/bart-base"


def train(config_path: str | Path = "configs/default.yaml") -> None:
    """Fine-tune an unmodified BART model."""
    config = load_yaml_config(config_path)
    trainer = build_trainer(config)
    trainer.train()
    _write_epoch_loss_report(trainer)
    if bool(config["training"].get("save_model", True)):
        trainer.save_model()


def build_trainer(config: dict[str, Any]) -> Seq2SeqTrainer:
    """Build a Hugging Face trainer from project config."""
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    seed = int(training_config.get("seed", 42))
    set_seed(seed)

    dataset_path = _require_file(data_config["dataset_path"], "dataset CSV")
    dataset = MentalHealthResponseDataset.from_csv(
        path=dataset_path,
        question_column=data_config.get("question_column", "question"),
        response_column=data_config.get("response_column", "response"),
        limit=data_config.get("max_examples"),
    )
    train_dataset, validation_dataset, _ = split_dataset(
        dataset,
        validation_size=float(data_config.get("validation_size", 0.1)),
        test_size=float(data_config.get("test_size", 0.1)),
        seed=seed,
    )

    model_name = model_config.get("name", DEFAULT_MODEL_NAME)
    local_files_only = bool(model_config.get("local_files_only", False))
    tokenizer = load_bart_tokenizer(
        model_name,
        local_files_only=local_files_only,
        add_prefix_space=bool(model_config.get("add_prefix_space", False)),
    )

    model = BartForConditionalGeneration.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )

    collator = BartDataCollator(
        tokenizer=tokenizer,
        max_source_length=int(data_config.get("max_source_length", 256)),
        max_target_length=int(data_config.get("max_target_length", 512)),
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_arguments = build_training_arguments(training_config)
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=int(training_config.get("early_stopping_patience", 2))
    )
    return Seq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[early_stopping],
    )


def build_training_arguments(training_config: dict[str, Any]) -> Seq2SeqTrainingArguments:
    """Create training arguments that work locally and on Kaggle."""
    require_cuda = bool(training_config.get("require_cuda", False))
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was required by config, but no GPU is available. "
            "On Kaggle, enable a GPU accelerator in notebook settings."
        )

    use_fp16 = bool(training_config.get("fp16", False)) and torch.cuda.is_available()
    return Seq2SeqTrainingArguments(
        output_dir=training_config.get("output_dir", "models/bart_baseline"),
        per_device_train_batch_size=int(training_config.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(training_config.get("per_device_eval_batch_size", 4)),
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 4)),
        learning_rate=float(training_config.get("learning_rate", 3e-5)),
        num_train_epochs=float(training_config.get("num_train_epochs", 40)),
        max_steps=int(training_config.get("max_steps", -1)),
        fp16=use_fp16,
        seed=int(training_config.get("seed", 42)),
        eval_strategy=training_config.get("eval_strategy", "epoch"),
        save_strategy=training_config.get("save_strategy", "epoch"),
        logging_strategy=training_config.get("logging_strategy", "epoch"),
        save_total_limit=int(training_config.get("save_total_limit", 2)),
        load_best_model_at_end=bool(training_config.get("load_best_model_at_end", True)),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(training_config.get("greater_is_better", False)),
        predict_with_generate=bool(training_config.get("predict_with_generate", False)),
        remove_unused_columns=False,
        report_to=training_config.get("report_to", "none"),
        optim=training_config.get("optim", "adamw_torch"),
        dataloader_num_workers=int(training_config.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(
            training_config.get("dataloader_pin_memory", torch.cuda.is_available())
        ),
    )


def _write_epoch_loss_report(trainer: Seq2SeqTrainer) -> Path:
    """Write one row per epoch containing the available train and evaluation losses."""
    losses_by_epoch: dict[float, dict[str, float]] = {}
    for entry in trainer.state.log_history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue

        epoch_value = float(epoch)
        row = losses_by_epoch.setdefault(epoch_value, {})
        if "loss" in entry:
            row["loss"] = float(entry["loss"])
        if "eval_loss" in entry:
            row["eval_loss"] = float(entry["eval_loss"])

    output_path = Path(trainer.args.output_dir) / "epoch_losses.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "loss", "eval_loss"])
        writer.writeheader()
        for epoch in sorted(losses_by_epoch):
            row = losses_by_epoch[epoch]
            writer.writerow(
                {
                    "epoch": epoch,
                    "loss": row.get("loss", ""),
                    "eval_loss": row.get("eval_loss", ""),
                }
            )

    return output_path


def _require_file(path: str | Path, label: str) -> Path:
    resolved_path = Path(path)
    if not resolved_path.exists():
        available_files = _format_available_kaggle_files()
        raise FileNotFoundError(f"Missing {label}: {resolved_path}{available_files}")

    return resolved_path


def _format_available_kaggle_files() -> str:
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return ""

    files = sorted(str(path) for path in kaggle_input.rglob("*.csv"))
    if not files:
        return "\nNo CSV files were found under /kaggle/input."

    visible_files = "\n".join(f"  - {path}" for path in files[:20])
    extra_count = len(files) - 20
    suffix = f"\n  ... and {extra_count} more CSV files" if extra_count > 0 else ""
    return f"\nAvailable CSV files under /kaggle/input:\n{visible_files}{suffix}"
