import pytest

from eat_bart.training.alpha_schedule import LinearAlphaWarmupCallback
from eat_bart.training.train import _build_callbacks
from eat_bart.utils.config import load_yaml_config


def test_kaggle_config_inherits_decoder_defaults() -> None:
    config = load_yaml_config("configs/kaggle.yaml")

    assert config["model"]["name"] == "facebook/bart-base"
    assert config["model"]["modify_encoder_self_attention"] is False
    assert config["model"]["modify_decoder_self_attention"] is True
    assert config["data"]["dataset_path"].endswith(
        "finalMentalHealthDataset-question-response.csv"
    )


def test_decoder_training_config_is_explicit_and_self_contained() -> None:
    config = load_yaml_config("configs/kaggle_decoder_eat.yaml")

    assert config["comparison"]["protocol_id"] == "bart_eat_comparable_v1"
    assert config["comparison"]["expected_cuda_devices"] == 2
    assert config["comparison"]["require_baseline_manifest"] is True
    assert config["comparison"]["allow_torch_version_drift"] is True
    assert config["model"]["baseline_artifact_name"] == "bart_baseline_comparable"
    assert config["model"]["emotion_model_name"] == (
        "SamLowe/roberta-base-go_emotions"
    )
    assert config["model"]["modify_encoder_self_attention"] is False
    assert config["model"]["modify_decoder_self_attention"] is True
    assert config["model"]["modify_cross_attention"] is False
    assert config["model"]["freeze_bart"] is True
    assert config["model"]["emotion_hidden_dim"] == 64
    assert config["model"]["emotion_weight_initialization"] == "tied_orthogonal"
    assert config["model"]["attention_formula"] == "probability_mix"
    assert config["model"]["alpha"] == pytest.approx(0.05)
    assert config["training"]["learning_rate"] == pytest.approx(1e-4)
    assert config["training"]["num_train_epochs"] == 40
    assert config["training"]["require_eat_weight_update"] is True
    assert config["training"]["early_stopping_patience"] is None
    assert config["data"]["max_source_length"] == 256
    assert config["data"]["max_target_length"] == 512
    assert "decoder" in config["data"]["contextual_emotion_cache"]["path"]

    callbacks = _build_callbacks(
        config["training"],
        alpha_target=config["model"]["alpha"],
        attention_formula=config["model"]["attention_formula"],
    )
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], LinearAlphaWarmupCallback)
    assert callbacks[0].start_alpha == pytest.approx(0.005)
    assert callbacks[0].target_alpha == pytest.approx(0.05)
    assert callbacks[0].warmup_epochs == pytest.approx(5)


def test_decoder_smoke_saves_checkpoint_and_checks_weight_update() -> None:
    config = load_yaml_config("configs/kaggle_decoder_eat_smoke.yaml")

    assert config["comparison"]["require_baseline_manifest"] is False
    assert config["training"]["max_steps"] == 5
    assert config["training"]["num_train_epochs"] == 1
    assert config["training"]["save_model"] is True
    assert config["training"]["require_eat_weight_update"] is True
    assert config["training"]["eval_strategy"] == "no"
    assert config["training"]["save_strategy"] == "no"
    assert config["data"]["max_examples"] == 64
    assert "smoke" in config["data"]["contextual_emotion_cache"]["path"]


def test_smoke_evaluations_cover_greedy_and_beam_generation() -> None:
    greedy = load_yaml_config(
        "configs/kaggle_decoder_eat_smoke_evaluate_greedy.yaml"
    )
    beam = load_yaml_config("configs/kaggle_decoder_eat_smoke_evaluate_beam.yaml")

    assert greedy["evaluation"]["checkpoint_path"] == (
        "/kaggle/working/models/decoder_eat_smoke"
    )
    assert greedy["evaluation"]["num_beams"] == 1
    assert greedy["evaluation"]["max_eval_examples"] == 8
    assert beam["evaluation"]["num_beams"] == 4
    assert beam["evaluation"]["output_path"] != greedy["evaluation"]["output_path"]


def test_full_decoder_evaluation_matches_baseline_generation_protocol() -> None:
    config = load_yaml_config("configs/kaggle_decoder_eat_evaluate.yaml")
    evaluation = config["evaluation"]

    assert evaluation["checkpoint_path"] == "/kaggle/working/models/decoder_eat"
    assert evaluation["max_eval_examples"] is None
    assert evaluation["max_new_tokens"] == 512
    assert evaluation["num_beams"] == 4
    assert evaluation["do_sample"] is False
    assert evaluation["repetition_penalty"] == pytest.approx(1.15)
    assert evaluation["no_repeat_ngram_size"] == 3
    assert evaluation["length_penalty"] == pytest.approx(1.15)
    assert evaluation["emotion_feature_condition"] == "aligned"


def test_decoder_evaluation_controls_are_isolated() -> None:
    alpha_zero = load_yaml_config("configs/kaggle_decoder_eat_alpha0_evaluate.yaml")
    shuffled = load_yaml_config("configs/kaggle_decoder_eat_shuffled_evaluate.yaml")
    zero = load_yaml_config("configs/kaggle_decoder_eat_zero_evaluate.yaml")
    alpha_zero_score = load_yaml_config("configs/kaggle_decoder_eat_alpha0_score.yaml")
    shuffled_score = load_yaml_config("configs/kaggle_decoder_eat_shuffled_score.yaml")

    assert alpha_zero["model"]["alpha"] == 0.0
    assert shuffled["evaluation"]["emotion_feature_condition"] == "shuffled"
    assert shuffled["evaluation"]["emotion_feature_shuffle_seed"] == 1729
    assert zero["evaluation"]["emotion_feature_condition"] == "zero"
    assert alpha_zero_score["scoring"]["validation_loss_path"] is None
    assert "validation_loss_path" not in shuffled_score["scoring"]


def test_decoder_judge_configs_use_decoder_artifacts() -> None:
    gpt_oss = load_yaml_config("configs/kaggle_decoder_eat_judge_gpt_oss.yaml")
    qwen = load_yaml_config("configs/kaggle_decoder_eat_judge_qwen.yaml")
    aggregate = load_yaml_config("configs/kaggle_decoder_eat_judge_aggregate.yaml")
    pairwise = load_yaml_config(
        "configs/kaggle_decoder_eat_pairwise_shuffled.yaml"
    )["pairwise_judge"]

    assert gpt_oss["llm_judge"]["model"] == "openai/gpt-oss-120b"
    assert gpt_oss["llm_judge"]["max_examples"] == 100
    assert qwen["llm_judge"]["model"] == "qwen/qwen3.8-27b"
    assert qwen["llm_judge"]["reasoning_effort"] == "none"
    assert aggregate["judge_aggregation"]["require_all_judges"] is True
    assert aggregate["judge_aggregation"]["min_judged_examples"] == 95
    assert pairwise["left_name"] == "shuffled_decoder_eat"
    assert pairwise["right_name"] == "aligned_decoder_eat"
