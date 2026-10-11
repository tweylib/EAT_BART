"""Causal contextual emotion features for BART decoder self-attention."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers.models.bart.modeling_bart import shift_tokens_right

from eat_bart.data.contextual_emotion import (
    AlignmentDiagnostics,
    align_hidden_states,
    tokenize_and_align_contextual_emotion,
)


def build_causal_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return a RoBERTa-compatible causal key mask.

    attention_mask shape: [batch_size, seq_len]
    return shape: [batch_size, 1, seq_len, seq_len]

    A query may attend only to valid keys at its own position or earlier. Padding
    queries retain access to earlier valid keys, but their output rows are never
    aligned to BART content tokens.
    """
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must have shape [batch_size, seq_len].")

    seq_len = attention_mask.size(1)
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    valid_keys = attention_mask.detach().cpu().bool().unsqueeze(1)
    return (causal.unsqueeze(0) & valid_keys).unsqueeze(1)


def causal_decoder_cache_fingerprint(
    texts: Sequence[str],
    model_name: str,
    max_length: int,
    add_prefix_space: bool,
    decoder_start_token_id: int,
) -> str:
    """Return a stable key for decoder-token-aligned causal features."""
    digest = hashlib.sha256()
    digest.update(
        (
            f"{model_name}\0{max_length}\0{add_prefix_space}\0"
            f"{decoder_start_token_id}\0causal_decoder_triangular_v1"
        ).encode()
    )
    for value in texts:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class CausalDecoderEmotionFeatureExtractor:
    """Build decoder features using a frozen, causally masked GoEmotions encoder.

    The input is the exact BART decoder prefix. It is decoded to canonical text,
    retokenized by BART and RoBERTa, and checked for a lossless BART-token
    round-trip before any feature is accepted. RoBERTa receives a triangular
    mask, so the representation aligned to decoder position ``t`` cannot depend
    on a token after ``t``.
    """

    def __init__(
        self,
        bart_tokenizer: Any,
        emotion_tokenizer: Any,
        emotion_model: torch.nn.Module,
        max_length: int,
        *,
        condition: str = "aligned",
        shuffle_seed: int = 1729,
    ) -> None:
        normalized_condition = str(condition).lower()
        if normalized_condition not in {"aligned", "shuffled", "zero"}:
            raise ValueError(
                "Decoder emotion condition must be 'aligned', 'shuffled', or 'zero'."
            )
        self.bart_tokenizer = bart_tokenizer
        self.emotion_tokenizer = emotion_tokenizer
        self.emotion_model = emotion_model.eval()
        self.emotion_model.requires_grad_(False)
        self.max_length = int(max_length)
        self.condition = normalized_condition
        self.shuffle_seed = int(shuffle_seed)

    def __call__(self, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        """Return features aligned to exact decoder IDs.

        decoder_input_ids shape: [batch_size, decoder_seq_len]
        return shape: [batch_size, decoder_seq_len, emotion_dim]
        """
        if decoder_input_ids.dim() != 2:
            raise ValueError(
                "decoder_input_ids must have shape [batch_size, decoder_seq_len]."
            )
        if decoder_input_ids.size(1) > self.max_length:
            raise ValueError(
                "Decoder prefix exceeds the configured causal emotion maximum length."
            )

        device = decoder_input_ids.device
        self.emotion_model.to(device)
        decoder_ids_cpu = decoder_input_ids.detach().cpu()
        texts = self.bart_tokenizer.batch_decode(
            decoder_ids_cpu,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        bart, emotion, alignment, diagnostics = tokenize_and_align_contextual_emotion(
            texts,
            self.bart_tokenizer,
            self.emotion_tokenizer,
            max_length=self.max_length,
            padding=True,
        )
        if diagnostics.alignment_failures:
            raise ValueError(
                "Causal decoder features failed to align "
                f"{diagnostics.alignment_failures} valid BART tokens."
            )

        causal_mask = build_causal_attention_mask(emotion["attention_mask"]).to(device)
        emotion_inputs = {
            key: value.to(device)
            for key, value in emotion.items()
            if key != "attention_mask"
        }
        with torch.no_grad():
            hidden = self.emotion_model(
                **emotion_inputs,
                attention_mask=causal_mask,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[-1]
            aligned = align_hidden_states(hidden, alignment.to(device))

        features, content_positions = self._map_to_decoder_ids(
            decoder_ids_cpu=decoder_ids_cpu,
            retokenized_ids=bart["input_ids"],
            retokenized_attention_mask=bart["attention_mask"],
            aligned=aligned,
        )
        return self._apply_condition(features, content_positions)

    def _map_to_decoder_ids(
        self,
        decoder_ids_cpu: torch.Tensor,
        retokenized_ids: torch.Tensor,
        retokenized_attention_mask: torch.Tensor,
        aligned: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Map canonical-text features back to the exact generated token positions."""
        special_ids = set(int(value) for value in self.bart_tokenizer.all_special_ids)
        batch_size, decoder_len = decoder_ids_cpu.shape
        emotion_dim = aligned.size(-1)
        features = aligned.new_zeros(batch_size, decoder_len, emotion_dim)
        all_content_positions: list[torch.Tensor] = []

        for row in range(batch_size):
            decoder_positions = torch.tensor(
                [
                    index
                    for index, token_id in enumerate(decoder_ids_cpu[row].tolist())
                    if int(token_id) not in special_ids
                ],
                dtype=torch.long,
            )
            target_positions = torch.tensor(
                [
                    index
                    for index, (token_id, valid) in enumerate(
                        zip(
                            retokenized_ids[row].tolist(),
                            retokenized_attention_mask[row].tolist(),
                            strict=True,
                        )
                    )
                    if valid and int(token_id) not in special_ids
                ],
                dtype=torch.long,
            )
            original_content = decoder_ids_cpu[row, decoder_positions]
            retokenized_content = retokenized_ids[row, target_positions]
            if not torch.equal(original_content, retokenized_content):
                raise ValueError(
                    "Generated BART prefix did not round-trip through canonical text; "
                    "refusing to guess decoder emotion alignment."
                )
            if decoder_positions.numel():
                features[row, decoder_positions.to(features.device)] = aligned[
                    row, target_positions.to(aligned.device)
                ]
            all_content_positions.append(decoder_positions)

        return features, all_content_positions

    def _apply_condition(
        self,
        features: torch.Tensor,
        content_positions: list[torch.Tensor],
    ) -> torch.Tensor:
        if self.condition == "aligned":
            return features
        if self.condition == "zero":
            return torch.zeros_like(features)

        # An ordinary full-sequence permutation would either expose future
        # information or change old positions whenever a prefix grows. This
        # causal control instead gives each content position a deterministic
        # feature from its strict history. The mapping is stable across prefix
        # extension and beam reordering; the first content feature is zero.
        shuffled = torch.zeros_like(features)
        for row, positions_cpu in enumerate(content_positions):
            positions = positions_cpu.to(features.device)
            for content_index in range(1, positions.numel()):
                generator = torch.Generator(device="cpu").manual_seed(
                    self.shuffle_seed + content_index
                )
                source_index = int(
                    torch.randint(0, content_index, (1,), generator=generator).item()
                )
                shuffled[row, positions[content_index]] = features[
                    row, positions[source_index]
                ]
        return shuffled


def load_or_build_causal_decoder_cache(
    texts: Sequence[str],
    bart_tokenizer: Any,
    emotion_tokenizer: Any,
    emotion_model: torch.nn.Module,
    model_name: str,
    cache_path: str | Path,
    max_length: int,
    decoder_start_token_id: int,
    pad_token_id: int,
    batch_size: int = 8,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load or build decoder-input-aligned causal features keyed by response text."""
    path = Path(cache_path)
    fingerprint = causal_decoder_cache_fingerprint(
        texts,
        model_name,
        max_length,
        bool(getattr(bart_tokenizer, "add_prefix_space", False)),
        decoder_start_token_id,
    )
    if path.exists():
        return load_causal_decoder_cache(path, fingerprint)

    path.parent.mkdir(parents=True, exist_ok=True)
    extractor = CausalDecoderEmotionFeatureExtractor(
        bart_tokenizer=bart_tokenizer,
        emotion_tokenizer=emotion_tokenizer,
        emotion_model=emotion_model,
        max_length=max_length,
    )
    features: dict[str, torch.Tensor] = {}
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        encoded = bart_tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[labels == pad_token_id] = -100
        decoder_input_ids = shift_tokens_right(
            labels,
            pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
        ).to(device)
        aligned = extractor(decoder_input_ids).to(dtype).cpu()
        lengths = decoder_input_ids.ne(pad_token_id).sum(dim=1).tolist()
        for text_value, row, length in zip(batch_texts, aligned, lengths, strict=True):
            features[text_value] = row[: int(length)].contiguous()

    torch.save(
        {
            "fingerprint": fingerprint,
            "model_name": model_name,
            "max_length": max_length,
            "dtype": str(dtype),
            "causal_definition": "triangular_roberta_decoder_v1",
            "features": features,
        },
        path,
    )
    return features


def load_causal_decoder_cache(
    cache_path: str | Path,
    expected_fingerprint: str,
) -> dict[str, torch.Tensor]:
    """Load a decoder cache only when its full preprocessing key matches."""
    path = Path(cache_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"Causal decoder emotion cache is stale or incompatible: {path}")
    if payload.get("causal_definition") != "triangular_roberta_decoder_v1":
        raise ValueError(f"Causal decoder emotion cache has an unknown definition: {path}")
    return payload["features"]


def validate_causal_mask_prefix_invariance(
    full_hidden_states: torch.Tensor,
    prefix_hidden_states: torch.Tensor,
    prefix_length: int,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    """Raise if a full causal pass disagrees with the same shorter prefix pass."""
    if prefix_length < 1:
        raise ValueError("prefix_length must be positive.")
    expected = full_hidden_states[:, :prefix_length]
    actual = prefix_hidden_states[:, :prefix_length]
    if not torch.allclose(expected, actual, atol=atol, rtol=rtol):
        maximum_error = float((expected - actual).abs().max().item())
        raise ValueError(
            "Causal emotion encoder failed prefix invariance; "
            f"maximum absolute error was {maximum_error:.8g}."
        )
