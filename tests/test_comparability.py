from __future__ import annotations

import json

import pytest

from eat_bart.training.comparability import (
    resolve_baseline_checkpoint,
    validate_baseline_manifest,
    write_run_manifest,
)


def _config() -> dict:
    return {
        "comparison": {"protocol_id": "bart_eat_comparable_v1"},
        "model": {"name": "facebook/bart-base", "add_prefix_space": False},
        "data": {
            "question_column": "question",
            "response_column": "response",
            "validation_size": 0.1,
            "test_size": 0.1,
            "max_source_length": 256,
            "max_target_length": 512,
            "max_examples": None,
        },
        "training": {
            "seed": 42,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "fp16": True,
        },
    }


def test_manifest_validates_identical_baseline_and_eat_protocol(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("question,response\nq,r\n", encoding="utf-8")
    checkpoint_path = tmp_path / "bart_baseline_comparable"
    write_run_manifest(_config(), dataset_path, checkpoint_path, stage="baseline")

    validate_baseline_manifest(checkpoint_path, _config(), dataset_path)


def test_manifest_rejects_target_length_mismatch(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("question,response\nq,r\n", encoding="utf-8")
    checkpoint_path = tmp_path / "bart_baseline_comparable"
    write_run_manifest(_config(), dataset_path, checkpoint_path, stage="baseline")
    eat_config = _config()
    eat_config["data"]["max_target_length"] = 128

    with pytest.raises(ValueError, match="data.max_target_length"):
        validate_baseline_manifest(checkpoint_path, eat_config, dataset_path)


def test_auto_discovery_requires_exactly_one_named_checkpoint(tmp_path) -> None:
    checkpoint_path = tmp_path / "uploaded" / "bart_baseline_comparable"
    checkpoint_path.mkdir(parents=True)
    (checkpoint_path / "config.json").write_text(json.dumps({}), encoding="utf-8")
    (checkpoint_path / "model.safetensors").write_bytes(b"weights")

    assert resolve_baseline_checkpoint("auto", kaggle_input_root=tmp_path) == checkpoint_path


def test_auto_discovery_rejects_ambiguous_checkpoints(tmp_path) -> None:
    for upload_name in ("first", "second"):
        checkpoint_path = tmp_path / upload_name / "bart_baseline_comparable"
        checkpoint_path.mkdir(parents=True)
        (checkpoint_path / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint_path / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(FileNotFoundError, match="found 2"):
        resolve_baseline_checkpoint("auto", kaggle_input_root=tmp_path)
