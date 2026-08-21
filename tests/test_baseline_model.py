import csv
from types import SimpleNamespace

from transformers import BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartAttention

from eat_bart.training.train import _write_epoch_loss_report, build_training_arguments


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


def test_training_arguments_select_best_eval_loss_and_log_each_epoch(tmp_path) -> None:
    arguments = build_training_arguments(
        {
            "output_dir": str(tmp_path),
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        }
    )

    assert arguments.logging_strategy.value == "epoch"
    assert arguments.load_best_model_at_end is True
    assert arguments.metric_for_best_model == "eval_loss"
    assert arguments.greater_is_better is False


def test_epoch_loss_report_combines_training_and_evaluation_logs(tmp_path) -> None:
    trainer = SimpleNamespace(
        args=SimpleNamespace(output_dir=str(tmp_path)),
        state=SimpleNamespace(
            log_history=[
                {"loss": 2.0, "epoch": 1.0},
                {"eval_loss": 1.8, "epoch": 1.0},
                {"loss": 1.5, "epoch": 2.0},
                {"eval_loss": 1.4, "epoch": 2.0},
            ]
        ),
    )

    output_path = _write_epoch_loss_report(trainer)

    with output_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows == [
        {"epoch": "1.0", "loss": "2.0", "eval_loss": "1.8"},
        {"epoch": "2.0", "loss": "1.5", "eval_loss": "1.4"},
    ]
