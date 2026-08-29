from pathlib import Path
import copy

import torch
from transformers import BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartAttention

from eat_bart.modeling.eat_bart_attention import EATBartAttention
from eat_bart.modeling.eat_attention import EATAttentionConfig
from eat_bart.modeling.eat_bart_model import (
    build_eat_bart_model_from_config,
    _validate_checkpoint_load_result,
)
from eat_bart.modeling.patch_bart import patch_bart_self_attention


def test_project_contract_preserves_cross_attention() -> None:
    brief = Path("PROJECT_BRIEF.md").read_text(encoding="utf-8")
    rules = Path("CODING_RULES.md").read_text(encoding="utf-8")

    assert "Do NOT modify cross-attention" in brief
    assert "Do not modify cross-attention" in rules


def test_patch_bart_self_attention_preserves_cross_attention() -> None:
    config = BartConfig(
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=99,
    )
    model = BartForConditionalGeneration(config)
    original_cross_attention = model.model.decoder.layers[0].encoder_attn

    patch_bart_self_attention(model)

    assert isinstance(model.model.encoder.layers[0].self_attn, EATBartAttention)
    assert isinstance(model.model.decoder.layers[0].self_attn, EATBartAttention)
    assert model.model.decoder.layers[0].encoder_attn is original_cross_attention
    assert isinstance(model.model.decoder.layers[0].encoder_attn, BartAttention)


def test_patch_bart_self_attention_can_leave_decoder_self_attention_unchanged() -> None:
    config = BartConfig(
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=99,
    )
    model = BartForConditionalGeneration(config)

    patch_bart_self_attention(model, modify_decoder_self_attention=False)

    assert isinstance(model.model.encoder.layers[0].self_attn, EATBartAttention)
    assert isinstance(model.model.decoder.layers[0].self_attn, BartAttention)
    assert not isinstance(model.model.decoder.layers[0].self_attn, EATBartAttention)


def test_probability_mix_alpha_zero_matches_unpatched_bart_logits_and_loss() -> None:
    torch.manual_seed(11)
    config = BartConfig(
        d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32, vocab_size=99,
        pad_token_id=1, bos_token_id=0, eos_token_id=2,
        decoder_start_token_id=2, dropout=0.0, attention_dropout=0.0,
        encoder_layerdrop=0.0, decoder_layerdrop=0.0,
    )
    config._attn_implementation = "sdpa"
    baseline = BartForConditionalGeneration(config).eval()
    eat_model = copy.deepcopy(baseline)
    patch_bart_self_attention(
        eat_model,
        EATAttentionConfig(2, 768, 8, 0.0, "probability_mix"),
        modify_encoder_self_attention=True,
        modify_decoder_self_attention=False,
    )
    eat_model.eval()
    input_ids = torch.tensor([[0, 5, 6, 7, 2], [0, 8, 9, 2, 1]])
    attention_mask = input_ids.ne(1).long()
    labels = torch.tensor([[10, 11, 2], [12, 13, 2]])
    emotion = torch.randn(2, 5, 768)
    with torch.no_grad():
        expected = baseline(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        actual = eat_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            aligned_emotion_hidden_states=emotion,
        )
    assert torch.allclose(actual.logits, expected.logits, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual.loss, expected.loss, atol=1e-7, rtol=1e-7)


def test_eat_wrapper_declares_that_it_does_not_handle_loss_kwargs() -> None:
    config = BartConfig(
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=99,
    )
    model = BartForConditionalGeneration(config)
    patch_bart_self_attention(model, modify_decoder_self_attention=False)

    assert model.accepts_loss_kwargs is False


def test_patched_bart_forward_accepts_encoder_and_decoder_emotion_features() -> None:
    config = BartConfig(
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=99,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
    )
    model = BartForConditionalGeneration(config)
    patch_bart_self_attention(model)

    input_ids = torch.tensor([[0, 5, 6, 2]])
    decoder_input_ids = torch.tensor([[2, 7, 8]])
    encoder_emotion_features = torch.zeros(1, 4, 8)
    decoder_emotion_features = torch.zeros(1, 3, 8)

    output = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        encoder_emotion_features=encoder_emotion_features,
        decoder_emotion_features=decoder_emotion_features,
    )

    assert tuple(output.logits.shape) == (1, 3, 99)


def test_encoder_emotion_features_backprop_to_encoder_alpha() -> None:
    config = BartConfig(
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=99,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
        dropout=0.0,
        attention_dropout=0.0,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
    )
    model = BartForConditionalGeneration(config)
    patch_bart_self_attention(model, modify_decoder_self_attention=False)

    input_ids = torch.tensor([[0, 5, 6, 7, 2]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[8, 9, 10, 2]])
    encoder_emotion_features = torch.tensor(
        [
            [
                [0.1, 0.0, 0.2, 0.0, 0.3, 0.0, 0.4, 0.0],
                [0.0, 0.2, 0.0, 0.3, 0.0, 0.4, 0.0, 0.5],
                [0.5, 0.0, 0.4, 0.0, 0.3, 0.0, 0.2, 0.0],
                [0.0, 0.4, 0.0, 0.2, 0.0, 0.1, 0.0, 0.3],
                [0.3, 0.1, 0.0, 0.2, 0.4, 0.0, 0.5, 0.0],
            ]
        ]
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        encoder_emotion_features=encoder_emotion_features,
    )
    output.loss.backward()

    alpha = model.model.encoder.layers[0].self_attn.emotion_interaction.alpha
    assert alpha.grad is not None
    assert alpha.grad.abs().sum().item() > 0.0


def test_encoder_emotion_features_support_padded_batches() -> None:
    config = BartConfig(
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=99,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
        dropout=0.0,
        attention_dropout=0.0,
    )
    model = BartForConditionalGeneration(config)
    patch_bart_self_attention(model, modify_decoder_self_attention=False)

    input_ids = torch.tensor([[0, 5, 6, 2, 1], [0, 7, 8, 9, 2]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    labels = torch.tensor([[10, 11, 2], [12, 13, 2]])
    encoder_emotion_features = torch.rand(2, 5, 8)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        encoder_emotion_features=encoder_emotion_features,
    )
    output.loss.backward()

    alpha = model.model.encoder.layers[0].self_attn.emotion_interaction.alpha
    assert torch.isfinite(output.loss)
    assert alpha.grad is not None
    assert torch.isfinite(alpha.grad).all()


def test_patched_bart_generate_accepts_precomputed_emotion_encoder_outputs() -> None:
    config = BartConfig(
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=99,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
        forced_eos_token_id=2,
    )
    model = BartForConditionalGeneration(config)
    patch_bart_self_attention(model)

    input_ids = torch.tensor([[0, 5, 6, 2]])
    attention_mask = torch.ones_like(input_ids)
    encoder_emotion_features = torch.zeros(1, 4, 8)
    encoder_outputs = model.model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_emotion_features=encoder_emotion_features,
    )

    generated = model.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        max_new_tokens=3,
        num_beams=1,
    )

    assert generated.size(0) == 1


def test_build_eat_bart_model_from_config_patches_self_attention() -> None:
    config = BartConfig(
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=99,
    )

    model = build_eat_bart_model_from_config(config)

    assert isinstance(model.model.encoder.layers[0].self_attn, EATBartAttention)
    assert isinstance(model.model.decoder.layers[0].self_attn, EATBartAttention)


def test_checkpoint_loader_allows_tied_weight_missing_aliases() -> None:
    _validate_checkpoint_load_result(
        missing_keys=[
            "model.encoder.embed_tokens.weight",
            "model.decoder.embed_tokens.weight",
            "lm_head.weight",
        ],
        unexpected_keys=[],
    )


def test_checkpoint_loader_rejects_unknown_missing_keys() -> None:
    try:
        _validate_checkpoint_load_result(missing_keys=["model.encoder.layers.0.fc1.weight"], unexpected_keys=[])
    except RuntimeError as error:
        assert "Unexpected missing checkpoint keys" in str(error)
    else:
        raise AssertionError("Expected unknown missing checkpoint keys to fail.")


def test_checkpoint_loader_rejects_unexpected_keys() -> None:
    try:
        _validate_checkpoint_load_result(missing_keys=[], unexpected_keys=["extra.weight"])
    except RuntimeError as error:
        assert "Unexpected checkpoint keys" in str(error)
    else:
        raise AssertionError("Expected unexpected checkpoint keys to fail.")
