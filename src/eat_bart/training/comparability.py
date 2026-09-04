"""Reproducibility and baseline/EAT compatibility safeguards."""

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


def resolve_baseline_checkpoint(
    configured_path: str | Path,
    artifact_name: str = "bart_baseline_comparable",
    kaggle_input_root: str | Path = "/kaggle/input",
) -> Path:
    """Resolve an explicit checkpoint or discover one unambiguously on Kaggle."""
    if str(configured_path) != "auto":
        path = Path(configured_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing baseline BART checkpoint: {path}")
        return path

    input_root = Path(kaggle_input_root)
    candidates = sorted(
        config_file.parent
        for config_file in input_root.rglob("config.json")
        if config_file.parent.name == artifact_name
        and any(
            (config_file.parent / weight_name).exists()
            for weight_name in ("model.safetensors", "pytorch_model.bin")
        )
    ) if input_root.exists() else []
    if len(candidates) != 1:
        formatted = "\n".join(f"  - {path}" for path in candidates) or "  (none)"
        raise FileNotFoundError(
            f"Expected exactly one uploaded '{artifact_name}' checkpoint under "
            f"{input_root}, found {len(candidates)}:\n{formatted}"
        )
    return candidates[0]


def protocol_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Extract settings that must be identical across the comparison stages."""
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


def validate_baseline_manifest(
    checkpoint_path: str | Path,
    eat_config: dict[str, Any],
    dataset_path: str | Path,
) -> None:
    """Fail before EAT training when its baseline protocol is incompatible."""
    manifest_path = Path(checkpoint_path) / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Comparable baseline checkpoint is missing {MANIFEST_FILENAME}: {checkpoint_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_signature = protocol_signature(eat_config)
    actual_signature = manifest.get("protocol_signature", {})
    mismatches = [
        f"{key}: baseline={actual_signature.get(key)!r}, EAT={value!r}"
        for key, value in expected_signature.items()
        if actual_signature.get(key) != value
    ]
    dataset_hash = _sha256_file(dataset_path)
    if manifest.get("dataset_sha256") != dataset_hash:
        mismatches.append(
            "dataset_sha256: baseline="
            f"{manifest.get('dataset_sha256')!r}, EAT={dataset_hash!r}"
        )
    baseline_software = manifest.get("software", {})
    current_software = {
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "accelerate": importlib.metadata.version("accelerate"),
        "tokenizers": importlib.metadata.version("tokenizers"),
    }
    for package, current_version in current_software.items():
        if baseline_software.get(package) != current_version:
            mismatches.append(
                f"software.{package}: baseline={baseline_software.get(package)!r}, "
                f"EAT={current_version!r}"
            )
    if mismatches:
        raise ValueError("Baseline/EAT protocol mismatch:\n  " + "\n  ".join(mismatches))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
