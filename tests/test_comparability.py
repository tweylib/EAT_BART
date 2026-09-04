from __future__ import annotations

import json

from eat_bart.training.comparability import protocol_signature, write_run_manifest


def test_manifest_records_protocol_and_dataset_identity(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("question,response\nq,r\n", encoding="utf-8")
    config = {
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

    output_path = write_run_manifest(config, dataset_path, tmp_path / "model", "baseline")
    manifest = json.loads(output_path.read_text(encoding="utf-8"))

    assert manifest["stage"] == "baseline"
    assert manifest["protocol_signature"] == protocol_signature(config)
    assert len(manifest["dataset_sha256"]) == 64


def test_protocol_signature_includes_target_length_and_batch_settings() -> None:
    config = {
        "comparison": {"protocol_id": "protocol"},
        "model": {},
        "data": {"max_target_length": 512},
        "training": {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
        },
    }
    signature = protocol_signature(config)

    assert signature["data.max_target_length"] == 512
    assert signature["training.per_device_train_batch_size"] == 4
    assert signature["training.gradient_accumulation_steps"] == 4
