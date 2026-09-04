"""Reproducibility manifest for comparable baseline/EAT runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import torch
import transformers

MANIFEST_FILENAME = "run_manifest.json"
PROTOCOL_FIELDS = (
    ("comparison", "protocol_id"),
    ("comparison", "expected_cuda_devices"),
    ("model", "name"),
    ("model", "add_prefix_space"),
    ("data", "question_column"),
    ("data", "response_column"),
    ("data", "validation_size"),
    ("data", "test_size"),
    ("data", "max_source_length"),
    ("data", "max_target_length"),
    ("data", "max_examples"),
    ("training", "seed"),
    ("training", "per_device_train_batch_size"),
    ("training", "per_device_eval_batch_size"),
    ("training", "gradient_accumulation_steps"),
    ("training", "fp16"),
)


def protocol_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Extract settings that must be identical across comparison stages."""
    signature: dict[str, Any] = {}
    for section, key in PROTOCOL_FIELDS:
        signature[f"{section}.{key}"] = config.get(section, {}).get(key)
    return signature


def validate_runtime(config: dict[str, Any]) -> None:
    """Require the GPU count declared by a controlled comparison config."""
    expected = config.get("comparison", {}).get("expected_cuda_devices")
    if expected is None:
        return
    actual = torch.cuda.device_count()
    if actual != int(expected):
        raise RuntimeError(
            f"Comparison protocol requires {expected} CUDA devices, found {actual}."
        )


def write_run_manifest(
    config: dict[str, Any],
    dataset_path: str | Path,
    output_dir: str | Path,
    stage: str,
) -> Path:
    """Persist the resolved protocol, dataset identity, and software versions."""
    output_path = Path(output_dir) / MANIFEST_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": stage,
        "protocol_signature": protocol_signature(config),
        "dataset_sha256": _sha256_file(dataset_path),
        "resolved_config": config,
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "accelerate": importlib.metadata.version("accelerate"),
            "tokenizers": importlib.metadata.version("tokenizers"),
        },
    }
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return output_path


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
