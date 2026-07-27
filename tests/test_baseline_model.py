from transformers import BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartAttention


def test_baseline_uses_stock_bart_attention_everywhere() -> None:
    config = BartConfig(
        vocab_size=64,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
    )
    model = BartForConditionalGeneration(config)

    encoder_layer = model.model.encoder.layers[0]
    decoder_layer = model.model.decoder.layers[0]

    assert type(encoder_layer.self_attn) is BartAttention
    assert type(decoder_layer.self_attn) is BartAttention
    assert type(decoder_layer.encoder_attn) is BartAttention
    assert not any("emotion" in name.lower() for name, _ in model.named_parameters())
