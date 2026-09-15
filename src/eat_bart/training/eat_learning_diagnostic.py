"""Diagnostics for learning and alignment in the encoder EAT branch."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from eat_bart.data.collator import EATBartDataCollator
from eat_bart.data.contextual_emotion import (
    contextual_cache_fingerprint,
    load_contextual_cache,
)
from eat_bart.data.dataset import MentalHealthResponseDataset, split_dataset
from eat_bart.data.tokenizer import load_bart_tokenizer
from eat_bart.modeling.eat_attention import EATAttentionConfig
from eat_bart.modeling.eat_bart_attention import EATBartAttention
from eat_bart.modeling.eat_bart_model import (
    load_eat_bart_checkpoint,
    load_eat_bart_from_baseline_checkpoint,
)
from eat_bart.training.comparability import resolve_baseline_checkpoint
from eat_bart.training.evaluate import _generate_rows
from eat_bart.training.train import _require_file
from eat_bart.utils.seed import set_seed


@dataclass
class _ProbabilityAccumulator:
    normalized_emotion_entropy: torch.Tensor
    emotion_kl_from_uniform: torch.Tensor
    emotion_standard_js: torch.Tensor
    emotion_standard_total_variation: torch.Tensor
    query_count: int = 0


def diagnose_eat_learning(config: dict[str, Any]) -> dict[str, Any]:
    """Run checkpoint, attention-distribution, shuffle, and generation diagnostics."""
    diagnostic_config = config["diagnostic"]
    model_config = config["model"]
    data_config = config["data"]
    evaluation_config = config["evaluation"]
    training_config = config.get("training", {})
    seed = int(training_config.get("seed", 42))
    set_seed(seed)

    checkpoint_dir = Path(diagnostic_config.get("checkpoint_dir", training_config["output_dir"]))
    best_checkpoint, final_checkpoint = resolve_diagnostic_checkpoints(checkpoint_dir)
    baseline_checkpoint = resolve_baseline_checkpoint(
        model_config["baseline_checkpoint_path"],
        artifact_name=model_config.get("baseline_artifact_name", "bart_baseline_comparable"),
    )

    dataset_path = _require_file(data_config["dataset_path"], "dataset CSV")
    dataset = MentalHealthResponseDataset.from_csv(
        path=dataset_path,
        question_column=data_config.get("question_column", "question"),
        response_column=data_config.get("response_column", "response"),
        limit=data_config.get("max_examples"),
    )
    _, validation_dataset, test_dataset = split_dataset(
        dataset,
        validation_size=float(data_config.get("validation_size", 0.1)),
        test_size=float(data_config.get("test_size", 0.1)),
        seed=seed,
    )

    tokenizer = load_bart_tokenizer(
        best_checkpoint,
        local_files_only=bool(model_config.get("local_files_only", False)),
        add_prefix_space=bool(model_config.get("add_prefix_space", True)),
    )
    cache_path = _require_file(
        data_config["contextual_emotion_cache"]["path"], "contextual emotion cache"
    )
    all_questions = [example.question for example in dataset.examples]
    fingerprint = contextual_cache_fingerprint(
        all_questions,
        model_config.get("emotion_model_name", "SamLowe/roberta-base-go_emotions"),
        int(data_config.get("max_source_length", 256)),
        bool(getattr(tokenizer, "add_prefix_space", False)),
    )
    contextual_cache = load_contextual_cache(cache_path, fingerprint)
    collator = EATBartDataCollator(
        tokenizer=tokenizer,
        lexicon={},
        max_source_length=int(data_config.get("max_source_length", 256)),
        max_target_length=int(data_config.get("max_target_length", 512)),
        emotion_feature_source="goemotions_contextual",
        contextual_emotion_cache=contextual_cache,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(diagnostic_config.get("batch_size", 4)),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    eat_config = EATAttentionConfig(
        num_heads=12,
        emotion_dim=int(model_config.get("emotion_dim", 768)),
        emotion_hidden_dim=int(model_config.get("emotion_hidden_dim", 32)),
        alpha_init=float(model_config.get("alpha", 0.05)),
        formula=model_config.get("attention_formula", "probability_mix"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_batches = int(diagnostic_config.get("max_attention_batches", 10))
    shuffle_seed = int(diagnostic_config.get("shuffle_seed", 1729))

    condition_summaries: dict[str, dict[str, float]] = {}
    attention_rows: list[dict[str, Any]] = []
    weights: dict[str, dict[tuple[int, int], torch.Tensor]] = {}
    generation_rows: dict[str, list[dict[str, str]]] = {}

    # Rebuild the original random EAT branch deterministically from the same
    # baseline and seed used by training. BART and the emotion model remain frozen.
    set_seed(seed)
    initial_model = load_eat_bart_from_baseline_checkpoint(
        baseline_checkpoint, eat_config=eat_config, local_files_only=True
    )
    weights["initial"] = extract_eat_head_weights(initial_model)
    summary, rows = measure_attention_and_loss(
        initial_model,
        validation_loader,
        device=device,
        condition="initial_aligned",
        max_batches=max_batches,
    )
    condition_summaries["initial_aligned"] = summary
    attention_rows.extend(rows)
    _release_model(initial_model)
    del initial_model

    best_model = load_eat_bart_checkpoint(
        best_checkpoint,
        eat_config=eat_config,
        modify_encoder_self_attention=True,
        modify_decoder_self_attention=False,
    )
    weights["best"] = extract_eat_head_weights(best_model)
    summary, rows = measure_attention_and_loss(
        best_model,
        validation_loader,
        device=device,
        condition="best_aligned",
        max_batches=max_batches,
    )
    condition_summaries["best_aligned"] = summary
    attention_rows.extend(rows)
    summary, rows = measure_attention_and_loss(
        best_model,
        validation_loader,
        device=device,
        condition="best_shuffled",
        max_batches=max_batches,
        feature_transform=lambda features, mask: shuffle_emotion_features(
            features, mask, seed=shuffle_seed
        ),
    )
    condition_summaries["best_shuffled"] = summary
    attention_rows.extend(rows)
    generation_rows["best"] = generate_diagnostic_rows(
        best_model,
        test_dataset,
        tokenizer=tokenizer,
        contextual_cache=contextual_cache,
        data_config=data_config,
        evaluation_config=evaluation_config,
        device=device,
        max_examples=int(diagnostic_config.get("generation_max_examples", 50)),
        batch_size=int(diagnostic_config.get("generation_batch_size", 4)),
    )
    _release_model(best_model)
    del best_model

    final_model = load_eat_bart_checkpoint(
        final_checkpoint,
        eat_config=eat_config,
        modify_encoder_self_attention=True,
        modify_decoder_self_attention=False,
    )
    weights["final"] = extract_eat_head_weights(final_model)
    summary, rows = measure_attention_and_loss(
        final_model,
        validation_loader,
        device=device,
        condition="final_aligned",
        max_batches=max_batches,
    )
    condition_summaries["final_aligned"] = summary
    attention_rows.extend(rows)
    generation_rows["final"] = generate_diagnostic_rows(
        final_model,
        test_dataset,
        tokenizer=tokenizer,
        contextual_cache=contextual_cache,
        data_config=data_config,
        evaluation_config=evaluation_config,
        device=device,
        max_examples=int(diagnostic_config.get("generation_max_examples", 50)),
        batch_size=int(diagnostic_config.get("generation_batch_size", 4)),
    )
    _release_model(final_model)
    del final_model

    weight_rows = calculate_weight_update_rows(weights)
    generation_comparison = compare_generation_rows(
        generation_rows["best"], generation_rows["final"]
    )
    condition_comparisons = compare_condition_summaries(condition_summaries)
    summary = {
        "baseline_checkpoint": str(baseline_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "seed": seed,
        "max_attention_batches": max_batches,
        "conditions": condition_summaries,
        "condition_comparisons": condition_comparisons,
        "weight_updates": summarize_weight_updates(weight_rows),
        "generation_comparison": generation_comparison[0],
    }

    output_dir = Path(diagnostic_config.get("output_dir", "/kaggle/working/reports"))
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = diagnostic_config.get(
        "output_prefix", "encoder_eat_comparable_a005_lr1e4_learning_diagnostic"
    )
    summary_path = output_dir / f"{prefix}.json"
    attention_path = output_dir / f"{prefix}_attention_heads.csv"
    weights_path = output_dir / f"{prefix}_weight_updates.csv"
    generations_path = output_dir / f"{prefix}_generation_comparison.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(attention_path, attention_rows)
    _write_csv(weights_path, weight_rows)
    _write_csv(generations_path, generation_comparison[1])

    print(json.dumps(summary, indent=2))
    print(f"Diagnostic summary: {summary_path}")
    print(f"Attention diagnostics: {attention_path}")
    print(f"Weight updates: {weights_path}")
    print(f"Generation comparison: {generations_path}")
    return summary


def resolve_diagnostic_checkpoints(checkpoint_dir: str | Path) -> tuple[Path, Path]:
    """Resolve the best and last checkpoints from retained Trainer states."""
    root = Path(checkpoint_dir)
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for state_path in root.glob("checkpoint-*/trainer_state.json"):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        candidates.append((int(state.get("global_step", 0)), state_path.parent, state))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint trainer_state.json files found under: {root}")

    _, final_checkpoint, final_state = max(candidates, key=lambda item: item[0])
    best_value = final_state.get("best_model_checkpoint")
    if not best_value:
        raise ValueError(
            "Final Trainer state does not identify a best checkpoint: "
            f"{final_checkpoint}"
        )
    best_checkpoint = root / Path(str(best_value)).name
    if not best_checkpoint.exists():
        raise FileNotFoundError(f"Best checkpoint is not retained under {root}: {best_checkpoint}")
    return best_checkpoint, final_checkpoint


def extract_eat_head_weights(model: torch.nn.Module) -> dict[tuple[int, int], torch.Tensor]:
    """Return flattened CPU W1/W2 vectors for every encoder layer and head."""
    result: dict[tuple[int, int], torch.Tensor] = {}
    for layer, encoder_layer in enumerate(model.model.encoder.layers):
        interaction = encoder_layer.self_attn.emotion_interaction
        for head in range(interaction.w1_s.size(0)):
            result[(layer, head)] = torch.cat(
                (
                    interaction.w1_s[head].detach().float().cpu().flatten(),
                    interaction.w2_s[head].detach().float().cpu().flatten(),
                )
            )
    return result


def calculate_weight_update_rows(
    weights: dict[str, dict[tuple[int, int], torch.Tensor]],
) -> list[dict[str, Any]]:
    """Calculate per-head movement from initialization, best, and final states."""
    keys = sorted(weights["initial"])
    rows = []
    for layer, head in keys:
        initial = weights["initial"][(layer, head)]
        best = weights["best"][(layer, head)]
        final = weights["final"][(layer, head)]
        rows.append(
            {
                "layer": layer,
                "head": head,
                "initial_norm": float(torch.linalg.vector_norm(initial).item()),
                "best_norm": float(torch.linalg.vector_norm(best).item()),
                "final_norm": float(torch.linalg.vector_norm(final).item()),
                "relative_initial_to_best": relative_parameter_update(initial, best),
                "relative_best_to_final": relative_parameter_update(best, final),
                "relative_initial_to_final": relative_parameter_update(initial, final),
            }
        )
    return rows


def relative_parameter_update(before: torch.Tensor, after: torch.Tensor) -> float:
    """Return ||after-before||_2 / ||before||_2."""
    denominator = torch.linalg.vector_norm(before)
    if float(denominator.item()) == 0.0:
        return float("inf")
    return float((torch.linalg.vector_norm(after - before) / denominator).item())


def summarize_weight_updates(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Return mean and median relative movement for each checkpoint interval."""
    summary: dict[str, float] = {}
    for key in (
        "relative_initial_to_best",
        "relative_best_to_final",
        "relative_initial_to_final",
    ):
        values = torch.tensor([float(row[key]) for row in rows], dtype=torch.float64)
        summary[f"mean_{key}"] = float(values.mean().item())
        summary[f"median_{key}"] = float(values.median().item())
        summary[f"max_{key}"] = float(values.max().item())
    return summary


def compare_condition_summaries(
    summaries: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Return signed deltas that answer the primary diagnostic questions."""
    initial = summaries["initial_aligned"]
    best = summaries["best_aligned"]
    shuffled = summaries["best_shuffled"]
    final = summaries["final_aligned"]
    result = {
        "best_loss_minus_initial_loss": (
            best["teacher_forced_loss_subset"] - initial["teacher_forced_loss_subset"]
        ),
        "final_loss_minus_best_loss": (
            final["teacher_forced_loss_subset"] - best["teacher_forced_loss_subset"]
        ),
        "shuffled_loss_minus_aligned_loss": (
            shuffled["teacher_forced_loss_subset"] - best["teacher_forced_loss_subset"]
        ),
    }
    for metric in (
        "mean_head_normalized_emotion_entropy",
        "mean_head_emotion_kl_from_uniform",
        "mean_head_emotion_standard_js",
        "mean_head_emotion_standard_total_variation",
    ):
        result[f"shuffled_minus_aligned_{metric}"] = shuffled[metric] - best[metric]
    return result


def shuffle_emotion_features(
    features: torch.Tensor, attention_mask: torch.Tensor, seed: int
) -> torch.Tensor:
    """Shuffle nonzero emotion-token vectors within each example deterministically."""
    shuffled = features.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    features_cpu = features.detach().cpu()
    mask_cpu = attention_mask.detach().cpu().bool()
    for row in range(features.size(0)):
        emotional = features_cpu[row].abs().sum(dim=-1).gt(0)
        indices = torch.nonzero(mask_cpu[row] & emotional, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        permutation = indices[torch.randperm(indices.numel(), generator=generator)]
        shuffled[row, indices.to(shuffled.device)] = features[
            row, permutation.to(features.device)
        ]
    return shuffled


def measure_attention_and_loss(
    model: torch.nn.Module,
    dataloader: DataLoader[Any],
    device: torch.device,
    condition: str,
    max_batches: int,
    feature_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Measure subset loss and attention-distribution structure for one condition."""
    model.to(device).eval()
    modules = {
        name: module
        for name, module in model.model.encoder.named_modules()
        if isinstance(module, EATBartAttention) and not module.is_decoder
    }
    accumulators = {
        name: _ProbabilityAccumulator(
            normalized_emotion_entropy=torch.zeros(module.num_heads, dtype=torch.float64),
            emotion_kl_from_uniform=torch.zeros(module.num_heads, dtype=torch.float64),
            emotion_standard_js=torch.zeros(module.num_heads, dtype=torch.float64),
            emotion_standard_total_variation=torch.zeros(
                module.num_heads, dtype=torch.float64
            ),
        )
        for name, module in modules.items()
    }
    current_mask: torch.Tensor | None = None

    def capture(
        name: str,
        module: EATBartAttention,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        emotion_features = getattr(module, "_eat_emotion_features", None)
        if hidden_states is None or emotion_features is None or current_mask is None:
            return
        hidden_shape = (*hidden_states.shape[:-1], module.num_heads, module.head_dim)
        query = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        standard_scores = torch.matmul(query, key.transpose(2, 3)) * module.scaling
        emotion_scores = module.emotion_interaction(emotion_features)
        statistics, count = probability_statistics(
            standard_scores, emotion_scores, current_mask
        )
        accumulator = accumulators[name]
        for metric, values in statistics.items():
            target = getattr(accumulator, metric)
            target += values.detach().cpu().to(torch.float64)
        accumulator.query_count += count

    handles = [
        module.register_forward_pre_hook(
            lambda module, args, kwargs, name=name: capture(name, module, args, kwargs),
            with_kwargs=True,
        )
        for name, module in modules.items()
    ]
    total_loss = 0.0
    total_target_tokens = 0
    num_examples = 0
    num_batches = 0
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                if batch_index >= max_batches:
                    break
                mask_cpu = batch["attention_mask"]
                if feature_transform is not None:
                    batch["aligned_emotion_hidden_states"] = feature_transform(
                        batch["aligned_emotion_hidden_states"], mask_cpu
                    )
                device_batch = {key: value.to(device) for key, value in batch.items()}
                current_mask = device_batch["attention_mask"]
                outputs = model(**device_batch)
                target_tokens = int(device_batch["labels"].ne(-100).sum().item())
                total_loss += float(outputs.loss.item()) * target_tokens
                total_target_tokens += target_tokens
                num_examples += int(device_batch["input_ids"].size(0))
                num_batches += 1
    finally:
        for handle in handles:
            handle.remove()

    rows: list[dict[str, Any]] = []
    totals = {
        "normalized_emotion_entropy": 0.0,
        "emotion_kl_from_uniform": 0.0,
        "emotion_standard_js": 0.0,
        "emotion_standard_total_variation": 0.0,
    }
    head_count = 0
    for name, accumulator in accumulators.items():
        if accumulator.query_count == 0:
            continue
        layer = _layer_index(name)
        for head in range(accumulator.normalized_emotion_entropy.numel()):
            row: dict[str, Any] = {"condition": condition, "layer": layer, "head": head}
            for metric in totals:
                value = float(getattr(accumulator, metric)[head].item()) / accumulator.query_count
                row[metric] = value
                totals[metric] += value
            rows.append(row)
            head_count += 1

    summary = {
        "teacher_forced_loss_subset": total_loss / total_target_tokens,
        "num_batches": float(num_batches),
        "num_examples": float(num_examples),
        "num_target_tokens": float(total_target_tokens),
    }
    for metric, total in totals.items():
        summary[f"mean_head_{metric}"] = total / head_count
    return summary, rows


def probability_statistics(
    standard_scores: torch.Tensor,
    emotion_scores: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], int]:
    """Return per-head sums of entropy, uniform KL, JS, and total variation."""
    mask = attention_mask.bool()
    key_mask = mask[:, None, None, :]
    standard = standard_scores.float().masked_fill(~key_mask, -torch.inf).softmax(dim=-1)
    emotion = emotion_scores.float().masked_fill(~key_mask, -torch.inf).softmax(dim=-1)
    epsilon = torch.finfo(torch.float32).tiny
    log_standard = standard.clamp_min(epsilon).log()
    log_emotion = emotion.clamp_min(epsilon).log()
    entropy = -(emotion * log_emotion).sum(dim=-1)
    key_counts = mask.sum(dim=-1).clamp_min(2).float()
    log_key_counts = key_counts.log()[:, None, None]
    normalized_entropy = entropy / log_key_counts
    kl_uniform = log_key_counts - entropy
    midpoint = 0.5 * (standard + emotion)
    log_midpoint = midpoint.clamp_min(epsilon).log()
    js = 0.5 * (
        (standard * (log_standard - log_midpoint)).sum(dim=-1)
        + (emotion * (log_emotion - log_midpoint)).sum(dim=-1)
    )
    total_variation = 0.5 * (standard - emotion).abs().sum(dim=-1)
    valid_queries = mask[:, None, :].to(normalized_entropy.dtype)
    statistics = {
        "normalized_emotion_entropy": (normalized_entropy * valid_queries).sum(dim=(0, 2)),
        "emotion_kl_from_uniform": (kl_uniform * valid_queries).sum(dim=(0, 2)),
        "emotion_standard_js": (js * valid_queries).sum(dim=(0, 2)),
        "emotion_standard_total_variation": (
            total_variation * valid_queries
        ).sum(dim=(0, 2)),
    }
    return statistics, int(mask.sum().item())


def generate_diagnostic_rows(
    model: torch.nn.Module,
    test_dataset: Any,
    tokenizer: Any,
    contextual_cache: dict[str, torch.Tensor],
    data_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    device: torch.device,
    max_examples: int,
    batch_size: int,
) -> list[dict[str, str]]:
    """Generate a fixed test subset using the experiment's decoding settings."""
    examples = [test_dataset[index] for index in range(min(len(test_dataset), max_examples))]
    local_evaluation_config = dict(evaluation_config)
    local_evaluation_config["batch_size"] = batch_size
    model.to(device).eval()
    return _generate_rows(
        examples=examples,
        tokenizer=tokenizer,
        lexicon={},
        model=model,
        device=device,
        data_config=data_config,
        evaluation_config=local_evaluation_config,
        use_emotion_encoder=True,
        emotion_feature_source="goemotions_contextual",
        contextual_cache=contextual_cache,
    )


def compare_generation_rows(
    best_rows: list[dict[str, str]], final_rows: list[dict[str, str]]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compare best- and final-checkpoint generated token sequences."""
    if len(best_rows) != len(final_rows):
        raise ValueError("Generation comparisons require equal row counts.")
    rows: list[dict[str, Any]] = []
    different = 0
    for index, (best, final) in enumerate(zip(best_rows, final_rows, strict=True)):
        if best["question"] != final["question"]:
            raise ValueError("Generation rows are not aligned by question.")
        same_tokens = best["generated_token_ids"] == final["generated_token_ids"]
        different += int(not same_tokens)
        rows.append(
            {
                "index": index,
                "question": best["question"],
                "reference_response": best["reference_response"],
                "best_response": best["generated_response"],
                "final_response": final["generated_response"],
                "same_token_sequence": same_tokens,
            }
        )
    count = len(rows)
    return (
        {
            "num_examples": float(count),
            "different_token_sequences": float(different),
            "different_token_sequence_rate": different / count if count else 0.0,
        },
        rows,
    )


def _layer_index(module_name: str) -> int:
    parts = module_name.split(".")
    if "layers" not in parts:
        raise ValueError(f"Could not determine encoder layer from module name: {module_name}")
    return int(parts[parts.index("layers") + 1])


def _release_model(model: torch.nn.Module) -> None:
    model.to("cpu")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty diagnostic CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
