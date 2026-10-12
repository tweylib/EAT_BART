"""Causal contextual emotion features for BART decoder self-attention."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers.models.bart.modeling_bart import shift_tokens_right


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
            f"{decoder_start_token_id}\0causal_decoder_token_map_v2"
        ).encode()
    )
    for value in texts:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class CausalDecoderEmotionFeatureExtractor:
    """Build decoder features using a frozen, causally masked GoEmotions encoder.

    BART and the GoEmotions RoBERTa use the same byte-BPE token strings. The
    exact live BART prefix is mapped token-by-token into the emotion tokenizer's
    vocabulary, avoiding the non-idempotent decode/retokenize cycle. RoBERTa
    receives a triangular mask, so the representation aligned to decoder
    position ``t`` cannot depend on a token after ``t``.
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
        self._emotion_id_by_bart_id = build_token_id_map(
            bart_tokenizer,
            emotion_tokenizer,
        )

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
        emotion_input_ids = self._map_bart_ids_to_emotion_ids(decoder_ids_cpu).to(device)
        pad_token_id = self.bart_tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Causal decoder features require a BART pad token ID.")
        attention_mask = decoder_ids_cpu.ne(int(pad_token_id)).to(torch.long)
        causal_mask = build_causal_attention_mask(attention_mask).to(device)
        with torch.no_grad():
            hidden = self.emotion_model(
                input_ids=emotion_input_ids,
                attention_mask=causal_mask,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[-1]

        special_ids = torch.tensor(
            list(self.bart_tokenizer.all_special_ids),
            dtype=decoder_ids_cpu.dtype,
        )
        is_special = torch.isin(decoder_ids_cpu, special_ids)
        is_content = attention_mask.bool() & ~is_special
        features = hidden.masked_fill(~is_content.to(device).unsqueeze(-1), 0.0)
        content_positions = [
            torch.nonzero(row, as_tuple=False).flatten() for row in is_content
        ]
        return self._apply_condition(features, content_positions)

    def _map_bart_ids_to_emotion_ids(
        self,
        decoder_ids_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Map exact BART IDs to equal token strings in the emotion vocabulary."""
        if decoder_ids_cpu.numel() == 0:
            return decoder_ids_cpu.clone()
        minimum_id = int(decoder_ids_cpu.min().item())
        maximum_id = int(decoder_ids_cpu.max().item())
        if minimum_id < 0 or maximum_id >= self._emotion_id_by_bart_id.numel():
            raise ValueError(
                "Decoder prefix contains a token ID outside the BART vocabulary: "
                f"min={minimum_id}, max={maximum_id}."
            )
        mapped = self._emotion_id_by_bart_id[decoder_ids_cpu]
        if bool(torch.any(mapped < 0)):
            missing_ids = torch.unique(decoder_ids_cpu[mapped < 0]).tolist()
            raise ValueError(
                "Decoder prefix contains BART tokens absent from the emotion "
                f"tokenizer vocabulary: {missing_ids[:8]}."
            )
        return mapped

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
            "causal_definition": "triangular_roberta_decoder_token_map_v2",
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
    if payload.get("causal_definition") != "triangular_roberta_decoder_token_map_v2":
        raise ValueError(f"Causal decoder emotion cache has an unknown definition: {path}")
    return payload["features"]


def build_token_id_map(
    bart_tokenizer: Any,
    emotion_tokenizer: Any,
) -> torch.Tensor:
    """Return a BART-ID to emotion-ID map based on exact token strings."""
    bart_vocab = bart_tokenizer.get_vocab()
    emotion_vocab = emotion_tokenizer.get_vocab()
    if not bart_vocab:
        raise ValueError("BART tokenizer vocabulary is empty.")

    mapping = torch.full(
        (max(int(value) for value in bart_vocab.values()) + 1,),
        -1,
        dtype=torch.long,
    )
    for token, bart_id in bart_vocab.items():
        emotion_id = emotion_vocab.get(token)
        if emotion_id is not None:
            mapping[int(bart_id)] = int(emotion_id)

    missing = int(torch.count_nonzero(mapping < 0).item())
    if missing:
        raise ValueError(
            "BART and emotion tokenizers do not share a complete token vocabulary; "
            f"{missing} BART token IDs cannot be mapped exactly."
        )
    return mapping


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
