"""Model factory for EAT-BART."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors_file
from transformers import BartConfig, BartForConditionalGeneration

from eat_bart.modeling.eat_attention import EATAttentionConfig
from eat_bart.modeling.patch_bart import patch_bart_self_attention

DEFAULT_MODEL_NAME = "facebook/bart-base"
EXPECTED_TIED_WEIGHT_MISSING_KEYS = {
    "model.encoder.embed_tokens.weight",
    "model.decoder.embed_tokens.weight",
    "lm_head.weight",
}


def build_eat_bart_model_from_config(
    config: BartConfig,
    eat_config: EATAttentionConfig | None = None,
    modify_encoder_self_attention: bool = True,
    modify_decoder_self_attention: bool = True,
) -> BartForConditionalGeneration:
    """Build an EAT-BART model from an in-memory BART config."""
    model = BartForConditionalGeneration(config)
    return patch_bart_self_attention(
        model,
        eat_config=eat_config,
        modify_encoder_self_attention=modify_encoder_self_attention,
        modify_decoder_self_attention=modify_decoder_self_attention,
    )


def load_eat_bart_model(
    model_name: str = DEFAULT_MODEL_NAME,
    eat_config: EATAttentionConfig | None = None,
    local_files_only: bool = False,
    modify_encoder_self_attention: bool = True,
    modify_decoder_self_attention: bool = True,
) -> BartForConditionalGeneration:
    """Load pretrained BART and apply EAT self-attention patches."""
    if model_name != DEFAULT_MODEL_NAME:
        raise ValueError("Changing the base model from facebook/bart-base requires explicit approval.")

    model = BartForConditionalGeneration.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    return patch_bart_self_attention(
        model,
        eat_config=eat_config,
        modify_encoder_self_attention=modify_encoder_self_attention,
        modify_decoder_self_attention=modify_decoder_self_attention,
    )


def load_eat_bart_from_baseline_checkpoint(
    checkpoint_path: str | Path,
    eat_config: EATAttentionConfig,
    local_files_only: bool = True,
) -> BartForConditionalGeneration:
    """Load a standard fine-tuned BART checkpoint, then add encoder-only EAT weights."""
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing baseline BART checkpoint: {checkpoint_dir}")
    model = BartForConditionalGeneration.from_pretrained(
        checkpoint_dir, local_files_only=local_files_only
    )
    return patch_bart_self_attention(
        model,
        eat_config=eat_config,
        modify_encoder_self_attention=True,
        modify_decoder_self_attention=False,
    )


def load_eat_bart_checkpoint(
    checkpoint_path: str | Path,
    eat_config: EATAttentionConfig | None = None,
    modify_encoder_self_attention: bool = True,
    modify_decoder_self_attention: bool = True,
) -> BartForConditionalGeneration:
    """Load a saved EAT-BART checkpoint."""
    checkpoint_dir = Path(checkpoint_path)
    config = BartConfig.from_pretrained(checkpoint_dir)
    model = build_eat_bart_model_from_config(
        config,
        eat_config=eat_config,
        modify_encoder_self_attention=modify_encoder_self_attention,
        modify_decoder_self_attention=modify_decoder_self_attention,
    )
    state_dict = _load_checkpoint_state_dict(checkpoint_dir)
    load_result = model.load_state_dict(state_dict, strict=False)
    _validate_checkpoint_load_result(load_result.missing_keys, load_result.unexpected_keys)
    model.tie_weights()
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


def _validate_checkpoint_load_result(
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> None:
    """Allow only known missing aliases from tied BART weights."""
    unexpected = sorted(unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")

    missing = set(missing_keys)
    unknown_missing = sorted(missing - EXPECTED_TIED_WEIGHT_MISSING_KEYS)
    if unknown_missing:
        raise RuntimeError(f"Unexpected missing checkpoint keys: {unknown_missing}")
