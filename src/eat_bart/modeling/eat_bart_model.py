"""Model factory for EAT-BART."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors_file
from transformers import BartConfig, BartForConditionalGeneration

from eat_bart.modeling.eat_attention import EATAttentionConfig
from eat_bart.modeling.patch_bart import patch_bart_self_attention

DEFAULT_MODEL_NAME = "facebook/bart-base"


def build_eat_bart_model_from_config(
    config: BartConfig,
    eat_config: EATAttentionConfig | None = None,
) -> BartForConditionalGeneration:
    """Build an EAT-BART model from an in-memory BART config."""
    model = BartForConditionalGeneration(config)
    return patch_bart_self_attention(model, eat_config=eat_config)


def load_eat_bart_model(
    model_name: str = DEFAULT_MODEL_NAME,
    eat_config: EATAttentionConfig | None = None,
    local_files_only: bool = False,
) -> BartForConditionalGeneration:
    """Load pretrained BART and apply EAT self-attention patches."""
    if model_name != DEFAULT_MODEL_NAME:
        raise ValueError("Changing the base model from facebook/bart-base requires explicit approval.")

    model = BartForConditionalGeneration.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    return patch_bart_self_attention(model, eat_config=eat_config)


def load_eat_bart_checkpoint(
    checkpoint_path: str | Path,
    eat_config: EATAttentionConfig | None = None,
) -> BartForConditionalGeneration:
    """Load a saved EAT-BART checkpoint."""
    checkpoint_dir = Path(checkpoint_path)
    config = BartConfig.from_pretrained(checkpoint_dir)
    model = build_eat_bart_model_from_config(config, eat_config=eat_config)
    state_dict = _load_checkpoint_state_dict(checkpoint_dir)
    model.load_state_dict(state_dict)
    return model


def _load_checkpoint_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    safetensors_path = checkpoint_dir / "model.safetensors"
    if safetensors_path.exists():
        return load_safetensors_file(safetensors_path)

    pytorch_path = checkpoint_dir / "pytorch_model.bin"
    if pytorch_path.exists():
        return torch.load(pytorch_path, map_location="cpu")

    raise FileNotFoundError(
        "Could not find model.safetensors or pytorch_model.bin in checkpoint directory: "
        f"{checkpoint_dir}"
    )
