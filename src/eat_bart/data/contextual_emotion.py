"""Tokenization and character-offset alignment for contextual emotion features."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class AlignmentDiagnostics:
    valid_bart_tokens: int
    aligned_bart_tokens: int
    alignment_failures: int
    used_identity_fast_path: bool

    @property
    def coverage(self) -> float:
        return self.aligned_bart_tokens / self.valid_bart_tokens if self.valid_bart_tokens else 1.0


def tokenize_and_align_contextual_emotion(
    texts: Sequence[str],
    bart_tokenizer: Any,
    emotion_tokenizer: Any,
    max_length: int,
    padding: bool | str = True,
) -> tuple[Any, Any, torch.Tensor, AlignmentDiagnostics]:
    """Tokenize raw baseline text and return RoBERTa-to-BART pooling weights.

    alignment shape: [batch_size, bart_seq_len, emotion_seq_len]. Every valid
    non-special BART token has a row summing to one. Special/padding rows are zero.
    """
    options = dict(max_length=max_length, padding=padding, truncation=True,
                   return_offsets_mapping=True, return_tensors="pt")
    bart = bart_tokenizer(list(texts), **options)
    emotion = emotion_tokenizer(list(texts), **options)
    bart_offsets = bart.pop("offset_mapping")
    emotion_offsets = emotion.pop("offset_mapping")
    alignment, diagnostics = build_offset_alignment(
        bart_offsets, emotion_offsets, bart["attention_mask"], emotion["attention_mask"]
    )
    return bart, emotion, alignment, diagnostics

def build_offset_alignment(
    bart_offsets: torch.Tensor,
    emotion_offsets: torch.Tensor,
    bart_attention_mask: torch.Tensor,
    emotion_attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, AlignmentDiagnostics]:
    """Build character-overlap mean-pooling weights with an exact-span fast path."""
    batch_size, bart_len, _ = bart_offsets.shape
    emotion_len = emotion_offsets.size(1)
    weights = torch.zeros(batch_size, bart_len, emotion_len, dtype=torch.float32)
    exact = bart_len == emotion_len and torch.equal(bart_offsets, emotion_offsets)
    valid = aligned = failures = 0

    for batch_index in range(batch_size):
        for bart_index, (start, end) in enumerate(bart_offsets[batch_index].tolist()):
            # Fast tokenizers encode special tokens as (0, 0); padding is also excluded.
            if not bart_attention_mask[batch_index, bart_index] or end <= start:
                continue
            valid += 1
            if exact and emotion_attention_mask[batch_index, bart_index]:
                weights[batch_index, bart_index, bart_index] = 1.0
                aligned += 1
                continue
            matches = []
            for emotion_index, (other_start, other_end) in enumerate(
                emotion_offsets[batch_index].tolist()
            ):
                if emotion_attention_mask[batch_index, emotion_index] and other_end > other_start:
                    if max(start, other_start) < min(end, other_end):
                        matches.append(emotion_index)
            if matches:
                weights[batch_index, bart_index, matches] = 1.0 / len(matches)
                aligned += 1
            else:
                failures += 1

    return weights, AlignmentDiagnostics(valid, aligned, failures, exact)


def align_hidden_states(hidden_states: torch.Tensor, alignment: torch.Tensor) -> torch.Tensor:
    """Pool RoBERTa states onto BART spans; result shape is [B, L_BART, 768]."""
    if alignment.size(0) != hidden_states.size(0) or alignment.size(2) != hidden_states.size(1):
        raise ValueError("Alignment dimensions do not match contextual emotion hidden states.")
    return torch.bmm(alignment.to(hidden_states.dtype), hidden_states)


def contextual_cache_fingerprint(
    texts: Sequence[str], model_name: str, max_length: int, add_prefix_space: bool
) -> str:
    """Return a stable key covering every input and tokenization-affecting setting."""
    digest = hashlib.sha256()
    # v3 restores the standard BART baseline's raw-text tokenization stream.
    digest.update(f"{model_name}\0{max_length}\0{add_prefix_space}\0v3".encode())
    for value in texts:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_or_build_contextual_cache(
    texts: Sequence[str],
    bart_tokenizer: Any,
    emotion_tokenizer: Any,
    emotion_model: torch.nn.Module,
    model_name: str,
    cache_path: str | Path,
    max_length: int,
    batch_size: int = 32,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load or precompute frozen, BART-aligned 768-d states keyed by exact text."""
    path = Path(cache_path)
    fingerprint = contextual_cache_fingerprint(
        texts, model_name, max_length, bool(getattr(bart_tokenizer, "add_prefix_space", False))
    )
    if path.exists():
        return load_contextual_cache(path, fingerprint)

    path.parent.mkdir(parents=True, exist_ok=True)
    emotion_model.to(device).eval()
    emotion_model.requires_grad_(False)
    features: dict[str, torch.Tensor] = {}
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        bart, emotion, alignment, diagnostics = tokenize_and_align_contextual_emotion(
            batch_texts, bart_tokenizer, emotion_tokenizer, max_length, padding=True
        )
        if diagnostics.alignment_failures:
            raise ValueError(
                f"Cache precomputation failed to align {diagnostics.alignment_failures} BART tokens."
            )
        emotion_inputs = {key: value.to(device) for key, value in emotion.items()}
        with torch.no_grad():
            hidden = emotion_model(
                **emotion_inputs, output_hidden_states=True, return_dict=True
            ).hidden_states[-1]
            aligned = align_hidden_states(hidden, alignment.to(device)).to(dtype).cpu()
        lengths = bart["attention_mask"].sum(dim=1).tolist()
        for text_value, row, length in zip(batch_texts, aligned, lengths, strict=True):
            features[text_value] = row[: int(length)].contiguous()

    torch.save(
        {"fingerprint": fingerprint, "model_name": model_name,
         "max_length": max_length, "dtype": str(dtype), "features": features},
        path,
    )
    return features


def load_contextual_cache(
    cache_path: str | Path, expected_fingerprint: str
) -> dict[str, torch.Tensor]:
    """Load a cache only when its complete preprocessing fingerprint matches."""
    path = Path(cache_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"Contextual emotion cache is stale or incompatible: {path}")
    return payload["features"]
