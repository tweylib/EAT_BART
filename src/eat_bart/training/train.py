"""Training loop setup for EAT-BART."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments

from eat_bart.data.collator import EATBartDataCollator
from eat_bart.data.contextual_emotion import load_or_build_contextual_cache
from eat_bart.data.dataset import MentalHealthResponseDataset, split_dataset
from eat_bart.data.emotion_lexicon import load_nrc_lexicon
from eat_bart.data.tokenizer import load_bart_tokenizer
from eat_bart.modeling.eat_attention import EATAttentionConfig
from eat_bart.modeling.eat_bart_model import (
    DEFAULT_MODEL_NAME,
    load_eat_bart_from_baseline_checkpoint,
    load_eat_bart_model,
)
from eat_bart.training.eat_signal import calculate_encoder_eat_signal, write_eat_signal_csv
from eat_bart.training.optimizer import DifferentialLearningRateTrainer
from eat_bart.utils.config import load_yaml_config
from eat_bart.utils.seed import set_seed


def train(config_path: str | Path = "configs/default.yaml") -> None:
    """Train EAT-BART."""
    config = load_yaml_config(config_path)
    trainer = build_trainer(config)
    trainer.train()
    if bool(config["training"].get("save_model", True)):
        trainer.save_model()
    _run_eat_signal_diagnostic(trainer, config.get("eat_signal"))


def build_trainer(config: dict[str, Any]) -> Seq2SeqTrainer:
    """Build a Hugging Face trainer from project config."""
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    seed = int(training_config.get("seed", 42))
    set_seed(seed)

    dataset_path = _require_file(data_config["dataset_path"], "dataset CSV")
    feature_source = model_config.get("emotion_feature_source", "lexicon")

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
    baseline_checkpoint_path = model_config.get("baseline_checkpoint_path")
    tokenizer_source = baseline_checkpoint_path or model_name
    tokenizer = load_bart_tokenizer(
        tokenizer_source,
        local_files_only=local_files_only,
        add_prefix_space=bool(model_config.get("add_prefix_space", True)),
    )

    lexicon = {}
    if feature_source == "lexicon":
        lexicon_path = _require_file(data_config["nrc_lexicon_path"], "NRC lexicon CSV")
        lexicon = load_nrc_lexicon(lexicon_path)
    eat_config = EATAttentionConfig(
        num_heads=12,
        emotion_dim=int(model_config.get("emotion_dim", 8)),
        emotion_hidden_dim=int(model_config.get("emotion_hidden_dim", 32)),
        alpha_init=float(model_config.get("alpha", model_config.get("alpha_init", 0.05))),
        formula=model_config.get("attention_formula", "additive"),
    )
    if baseline_checkpoint_path:
        model = load_eat_bart_from_baseline_checkpoint(
            baseline_checkpoint_path, eat_config=eat_config, local_files_only=True
        )
    else:
        model = load_eat_bart_model(
            model_name=model_name, eat_config=eat_config, local_files_only=local_files_only,
            modify_encoder_self_attention=bool(model_config.get("modify_encoder_self_attention", True)),
            modify_decoder_self_attention=bool(model_config.get("modify_decoder_self_attention", True)),
        )

    emotion_tokenizer = None
    contextual_cache = None
    if feature_source == "goemotions_contextual":
        emotion_model_name = model_config.get("emotion_model_name", "SamLowe/roberta-base-go_emotions")
        emotion_tokenizer = AutoTokenizer.from_pretrained(
            emotion_model_name, local_files_only=local_files_only, use_fast=True,
            add_prefix_space=bool(model_config.get("add_prefix_space", True)),
        )
        emotion_model = AutoModelForSequenceClassification.from_pretrained(
            emotion_model_name, local_files_only=local_files_only
        )
        emotion_model.eval()
        emotion_model.requires_grad_(False)
        emotion_hidden_size = int(emotion_model.config.hidden_size)
        configured_emotion_dim = int(model_config.get("emotion_dim", emotion_hidden_size))
        if configured_emotion_dim != emotion_hidden_size:
            raise ValueError(
                "Without W_E, model.emotion_dim must equal the GoEmotions hidden size "
                f"({emotion_hidden_size}), got {configured_emotion_dim}."
            )
        cache_config = data_config.get("contextual_emotion_cache", {})
        if bool(cache_config.get("enabled", False)):
            all_questions = [example.question for example in dataset.examples]
            cache_device = "cuda" if torch.cuda.is_available() else "cpu"
            contextual_cache = load_or_build_contextual_cache(
                texts=all_questions, bart_tokenizer=tokenizer,
                emotion_tokenizer=emotion_tokenizer, emotion_model=emotion_model,
                model_name=emotion_model_name, cache_path=cache_config["path"],
                max_length=int(data_config.get("max_source_length", 256)),
                batch_size=int(cache_config.get("batch_size", 32)),
                dtype=torch.float16, device=cache_device,
            )
            del emotion_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            model.contextual_emotion_encoder = emotion_model
        if bool(model_config.get("freeze_bart", False)):
            freeze_bart_for_eat_only_training(model)

    collator = EATBartDataCollator(
        tokenizer=tokenizer,
        lexicon=lexicon,
        max_source_length=int(data_config.get("max_source_length", 256)),
        max_target_length=int(data_config.get("max_target_length", 128)),
        subword_strategy=data_config.get("subword_strategy", "single"),
        decoder_start_token_id=model.config.decoder_start_token_id,
        emotion_feature_source=feature_source,
        emotion_tokenizer=emotion_tokenizer,
        contextual_emotion_cache=contextual_cache,
    )

    training_arguments = build_training_arguments(training_config)
    callbacks = _build_callbacks(training_config)
    trainer_class: type[Seq2SeqTrainer] = Seq2SeqTrainer
    trainer_kwargs: dict[str, Any] = {}
    if "eat_learning_rate" in training_config or "alpha_learning_rate" in training_config:
        trainer_class = DifferentialLearningRateTrainer
        base_learning_rate = float(training_config.get("learning_rate", 3e-5))
        trainer_kwargs["eat_learning_rate"] = float(
            training_config.get("eat_learning_rate", base_learning_rate)
        )
        if model_config.get("attention_formula", "additive") != "probability_mix":
            trainer_kwargs["alpha_learning_rate"] = float(
                training_config.get("alpha_learning_rate", base_learning_rate)
            )

    return trainer_class(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
        **trainer_kwargs,
    )


def freeze_bart_for_eat_only_training(model: torch.nn.Module) -> None:
    """Freeze BART and leave only encoder EAT W1 and W2 trainable."""
    model.requires_grad_(False)
    bart_model = getattr(model, "model", model)
    for layer in bart_model.encoder.layers:
        interaction = getattr(layer.self_attn, "emotion_interaction", None)
        if interaction is None:
            raise ValueError("Every patched encoder layer must contain emotion_interaction.")
        interaction.w1_s.requires_grad_(True)
        interaction.w2_s.requires_grad_(True)


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
        output_dir=training_config.get("output_dir", "models/eat_bart"),
        per_device_train_batch_size=int(training_config.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(training_config.get("per_device_eval_batch_size", 4)),
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 4)),
        learning_rate=float(training_config.get("learning_rate", 3e-5)),
        num_train_epochs=float(training_config.get("num_train_epochs", 2)),
        max_steps=int(training_config.get("max_steps", -1)),
        fp16=use_fp16,
        seed=int(training_config.get("seed", 42)),
        eval_strategy=training_config.get("eval_strategy", "epoch"),
        save_strategy=training_config.get("save_strategy", "epoch"),
        logging_steps=int(training_config.get("logging_steps", 50)),
        save_total_limit=int(training_config.get("save_total_limit", 2)),
        predict_with_generate=bool(training_config.get("predict_with_generate", False)),
        remove_unused_columns=False,
        report_to=training_config.get("report_to", "none"),
        optim=training_config.get("optim", "adamw_torch"),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
        max_grad_norm=float(training_config.get("max_grad_norm", 1.0)),
        dataloader_num_workers=int(training_config.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(
            training_config.get("dataloader_pin_memory", torch.cuda.is_available())
        ),
        load_best_model_at_end=bool(training_config.get("load_best_model_at_end", False)),
        metric_for_best_model=training_config.get("metric_for_best_model"),
        greater_is_better=training_config.get("greater_is_better"),
    )


def _build_callbacks(training_config: dict[str, Any]) -> list[EarlyStoppingCallback]:
    """Build optional validation-based early stopping callbacks."""
    patience = training_config.get("early_stopping_patience")
    if patience is None:
        return []

    patience = int(patience)
    if patience < 1:
        raise ValueError("early_stopping_patience must be at least 1.")
    if training_config.get("eval_strategy", "epoch") == "no":
        raise ValueError("Early stopping requires evaluation to be enabled.")
    if not bool(training_config.get("load_best_model_at_end", False)):
        raise ValueError("Early stopping requires load_best_model_at_end: true.")

    return [
        EarlyStoppingCallback(
            early_stopping_patience=patience,
            early_stopping_threshold=float(
                training_config.get("early_stopping_threshold", 0.0)
            ),
        )
    ]


def _run_eat_signal_diagnostic(
    trainer: Seq2SeqTrainer,
    diagnostic_config: dict[str, Any] | None,
) -> None:
    """Calculate and persist r_h for the best model after training."""
    if not diagnostic_config or not bool(diagnostic_config.get("enabled", True)):
        return

    rows, summary = calculate_encoder_eat_signal(
        model=trainer.model,
        dataloader=trainer.get_eval_dataloader(),
        max_batches=diagnostic_config.get("max_batches"),
    )
    output_path = write_eat_signal_csv(diagnostic_config["output_path"], rows)
    print(f"EAT signal report: {output_path}")
    for name, value in summary.items():
        print(f"{name}: {value:.8f}")


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
