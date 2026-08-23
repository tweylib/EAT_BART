"""Forward-pass shims that route emotion features through native BART."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from transformers.models.bart.modeling_bart import (
    BartEncoder,
    BartForConditionalGeneration,
)

from eat_bart.modeling.eat_bart_attention import EATBartAttention
from eat_bart.data.contextual_emotion import align_hidden_states


def enable_eat_forwarding(model: object) -> object:
    """Install EAT-aware entry points without replacing BART's internal forward logic."""
    if isinstance(model, BartForConditionalGeneration) and not isinstance(
        model, EATBartForConditionalGeneration
    ):
        model.__class__ = EATBartForConditionalGeneration

    bart_model = getattr(model, "model", model)
    encoder = getattr(bart_model, "encoder", None)
    if isinstance(encoder, BartEncoder) and not isinstance(encoder, EATBartEncoder):
        encoder.__class__ = EATBartEncoder

    return model


class EATBartForConditionalGeneration(BartForConditionalGeneration):
    """BART LM entry point that makes batch emotion features available to EAT attention."""

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_outputs: Any | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        encoder_emotion_features: torch.Tensor | None = None,
        decoder_emotion_features: torch.Tensor | None = None,
        emotion_input_ids: torch.LongTensor | None = None,
        emotion_attention_mask: torch.Tensor | None = None,
        bart_to_emotion_alignment: torch.Tensor | None = None,
        aligned_emotion_hidden_states: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if aligned_emotion_hidden_states is not None:
            # Cached states are already BART-aligned E in R^[B,L,768].
            encoder_emotion_features = aligned_emotion_hidden_states
        elif emotion_input_ids is not None:
            extractor = getattr(self, "contextual_emotion_encoder", None)
            if extractor is None or bart_to_emotion_alignment is None:
                raise ValueError("Contextual emotion inputs require encoder and alignment.")
            extractor.eval()
            with torch.no_grad():
                outputs = extractor(
                    input_ids=emotion_input_ids,
                    attention_mask=emotion_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                contextual = outputs.hidden_states[-1]
                aligned = align_hidden_states(contextual, bart_to_emotion_alignment)
            encoder_emotion_features = aligned

        with _route_emotion_features(
            self,
            encoder_emotion_features=encoder_emotion_features,
            decoder_emotion_features=decoder_emotion_features,
        ):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                **kwargs,
            )


class EATBartEncoder(BartEncoder):
    """BART encoder entry point used when generation precomputes encoder outputs."""

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_emotion_features: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if encoder_emotion_features is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        with _route_emotion_features(
            self,
            encoder_emotion_features=encoder_emotion_features,
        ):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )


@contextmanager
def _route_emotion_features(
    module: torch.nn.Module,
    encoder_emotion_features: torch.Tensor | None = None,
    decoder_emotion_features: torch.Tensor | None = None,
) -> Iterator[None]:
    """Temporarily attach each batch's features to its EAT attention modules."""
    previous_values: list[tuple[EATBartAttention, bool, torch.Tensor | None]] = []
    for attention in module.modules():
        if not isinstance(attention, EATBartAttention):
            continue

        features = (
            decoder_emotion_features if attention.is_decoder else encoder_emotion_features
        )
        if features is None:
            continue

        had_value = hasattr(attention, "_eat_emotion_features")
        previous_value = getattr(attention, "_eat_emotion_features", None)
        previous_values.append((attention, had_value, previous_value))
        attention._eat_emotion_features = features

    try:
        yield
    finally:
        for attention, had_value, previous_value in previous_values:
            if had_value:
                attention._eat_emotion_features = previous_value
            else:
                del attention._eat_emotion_features
