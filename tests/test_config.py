from copy import deepcopy

from eat_bart.utils.config import load_yaml_config
from eat_bart.training.train import _build_callbacks, build_training_arguments


def test_kaggle_config_inherits_default_config() -> None:
    config = load_yaml_config("configs/kaggle.yaml")

    assert config["model"]["name"] == "facebook/bart-base"
    assert config["data"]["dataset_path"].endswith("finalMentalHealthDataset-question-response.csv")
    assert config["training"]["output_dir"] == "/kaggle/working/models/eat_bart"


def test_kaggle_encoder_only_config_disables_decoder_self_attention_patch() -> None:
    config = load_yaml_config("configs/kaggle_encoder_only.yaml")

    assert config["model"]["modify_encoder_self_attention"] is True
    assert config["model"]["modify_decoder_self_attention"] is False
    assert config["training"]["output_dir"] == "/kaggle/working/models/eat_bart_encoder_only"


def test_kaggle_encoder_only_experiment_evaluate_uses_full_test_split() -> None:
    config = load_yaml_config("configs/kaggle_encoder_only_5epoch_experiment_evaluate.yaml")

    assert config["model"]["modify_decoder_self_attention"] is False
    assert config["evaluation"]["checkpoint_path"] == "/kaggle/working/models/eat_bart_encoder_only_5epoch"
    assert config["evaluation"]["max_eval_examples"] is None


def test_kaggle_encoder_only_experiment_uses_weighted_two_judge_aggregate() -> None:
    config = load_yaml_config(
        "configs/kaggle_encoder_only_5epoch_experiment_judge_groq_2judge_aggregate.yaml"
    )

    assert config["judge_aggregation"]["min_judged_examples"] == 1
    assert len(config["judge_aggregation"]["judges"]) == 2


def test_kaggle_encoder_only_alpha_0_1_experiment_is_isolated() -> None:
    training_config = load_yaml_config(
        "configs/kaggle_encoder_only_alpha_0_1_5epoch.yaml"
    )
    evaluation_config = load_yaml_config(
        "configs/kaggle_encoder_only_alpha_0_1_5epoch_experiment_evaluate.yaml"
    )

    assert training_config["model"]["alpha_init"] == 0.1
    assert training_config["model"]["modify_encoder_self_attention"] is True
    assert training_config["model"]["modify_decoder_self_attention"] is False
    assert training_config["training"]["num_train_epochs"] == 5
    assert training_config["training"]["seed"] == 42
    assert training_config["training"]["output_dir"].endswith(
        "eat_bart_encoder_only_alpha_0_1_5epoch"
    )
    assert evaluation_config["model"]["alpha_init"] == 0.1
    assert evaluation_config["evaluation"]["checkpoint_path"] == training_config[
        "training"
    ]["output_dir"]
    assert "alpha_0_1" in evaluation_config["evaluation"]["output_path"]


def test_differential_lr_early_stopping_experiment_protocol() -> None:
    config = load_yaml_config(
        "configs/kaggle_encoder_only_alpha_0_1_differential_lr_early_stop.yaml"
    )

    assert config["model"]["alpha_init"] == 0.1
    assert config["model"]["modify_encoder_self_attention"] is True
    assert config["model"]["modify_decoder_self_attention"] is False
    assert config["training"]["num_train_epochs"] == 40
    assert config["training"]["learning_rate"] == 1e-5
    assert config["training"]["eat_learning_rate"] == 5e-5
    assert config["training"]["alpha_learning_rate"] == 0.01
    assert config["training"]["early_stopping_patience"] == 2
    assert config["training"]["load_best_model_at_end"] is True
    assert config["training"]["metric_for_best_model"] == "eval_loss"
    assert config["training"]["greater_is_better"] is False
    assert config["eat_signal"]["max_batches"] == 50

    local_training_config = dict(config["training"], require_cuda=False)
    arguments = build_training_arguments(local_training_config)
    callbacks = _build_callbacks(config["training"])
    assert arguments.load_best_model_at_end is True
    assert arguments.metric_for_best_model == "eval_loss"
    assert arguments.greater_is_better is False
    assert len(callbacks) == 1
    assert callbacks[0].early_stopping_patience == 2


def test_contextual_eat_uses_standard_baseline_tokenization_contract() -> None:
    training_config = load_yaml_config(
        "configs/encoder_eat_goemotions_probmix_a010_full.yaml"
    )
    evaluation_config = load_yaml_config(
        "configs/encoder_eat_goemotions_probmix_a010_full_experiment_evaluate.yaml"
    )

    assert training_config["model"]["add_prefix_space"] is False
    assert training_config["data"]["max_source_length"] == 256
    assert training_config["data"]["max_target_length"] == 512
    assert "raw_aligned" in training_config["data"]["contextual_emotion_cache"]["path"]
    assert evaluation_config["evaluation"]["max_new_tokens"] == 512


def test_comparable_eat_protocol_is_explicit_and_self_checking() -> None:
    training_config = load_yaml_config("configs/kaggle_encoder_eat_comparable.yaml")
    evaluation_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_evaluate.yaml"
    )
    judge_config = load_yaml_config("configs/kaggle_encoder_eat_comparable_judge_gpt_oss.yaml")
    qwen_config = load_yaml_config("configs/kaggle_encoder_eat_comparable_judge_qwen.yaml")
    aggregate_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_judge_aggregate.yaml"
    )

    assert training_config["comparison"]["protocol_id"] == "bart_eat_comparable_v1"
    assert training_config["comparison"]["expected_cuda_devices"] == 2
    assert training_config["comparison"]["require_baseline_manifest"] is True
    assert training_config["model"]["baseline_checkpoint_path"] == (
        "/kaggle/input/datasets/cheikhtidjanitweylib/"
        "baseline-30-eps-new-model/models/bart_baseline_comparable"
    )
    assert training_config["model"]["baseline_artifact_name"] == "bart_baseline_comparable"
    assert training_config["model"]["add_prefix_space"] is False
    assert training_config["data"]["max_source_length"] == 256
    assert training_config["data"]["max_target_length"] == 512
    assert training_config["training"]["per_device_train_batch_size"] == 4
    assert training_config["training"]["per_device_eval_batch_size"] == 4
    assert training_config["training"]["gradient_accumulation_steps"] == 4
    assert training_config["training"]["early_stopping_patience"] == 3
    assert evaluation_config["evaluation"]["max_new_tokens"] == 512
    assert evaluation_config["evaluation"]["num_beams"] == 4
    assert evaluation_config["evaluation"]["repetition_penalty"] == 1.15
    assert evaluation_config["evaluation"]["no_repeat_ngram_size"] == 3
    assert evaluation_config["evaluation"]["length_penalty"] == 1.15
    assert judge_config["llm_judge"]["model"] == "openai/gpt-oss-120b"
    assert judge_config["llm_judge"]["max_examples"] == 100
    assert qwen_config["llm_judge"]["model"] == "qwen/qwen3.6-27b"
    assert qwen_config["llm_judge"]["max_examples"] == 100
    assert qwen_config["llm_judge"]["reasoning_effort"] == "none"
    assert qwen_config["llm_judge"]["response_format_json"] is True
    assert aggregate_config["judge_aggregation"]["require_all_judges"] is True
    assert aggregate_config["judge_aggregation"]["min_judged_examples"] == 95


def test_comparable_alpha_zero_ablation_is_evaluation_only() -> None:
    evaluation_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_alpha0_evaluate.yaml"
    )
    scoring_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_alpha0_score.yaml"
    )

    assert evaluation_config["model"]["alpha"] == 0.0
    assert evaluation_config["evaluation"]["model_source"] == "eat_checkpoint"
    assert evaluation_config["evaluation"]["checkpoint_path"] == (
        "/kaggle/input/datasets/cheikhmohamedahid/"
        "eat-encoder/models/encoder_eat_comparable"
    )
    assert evaluation_config["data"]["contextual_emotion_cache"]["path"] == (
        "/kaggle/input/datasets/cheikhmohamedahid/eat-encoder/cache/"
        "goemotions_baseline_raw_aligned_fp16_v3.pt"
    )
    assert evaluation_config["evaluation"]["do_sample"] is False
    assert scoring_config["scoring"]["validation_loss_path"] is None


def test_comparable_low_lr_diagnostic_changes_only_intended_settings() -> None:
    baseline_config = load_yaml_config("configs/kaggle_encoder_eat_comparable.yaml")
    training_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a010_lr1e4.yaml"
    )
    reference_evaluation_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_evaluate.yaml"
    )
    evaluation_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a010_lr1e4_evaluate.yaml"
    )
    scoring_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a010_lr1e4_score.yaml"
    )
    gpt_oss_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a010_lr1e4_judge_gpt_oss.yaml"
    )
    qwen_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a010_lr1e4_judge_qwen.yaml"
    )
    aggregate_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a010_lr1e4_judge_aggregate.yaml"
    )

    assert training_config["model"]["alpha"] == 0.10
    assert training_config["training"]["learning_rate"] == 0.0001
    for key in (
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "early_stopping_patience",
        "early_stopping_threshold",
    ):
        assert training_config["training"][key] == baseline_config["training"][key]
    assert training_config["data"]["max_source_length"] == 256
    assert training_config["data"]["max_target_length"] == 512
    assert training_config["data"]["contextual_emotion_cache"]["path"] == (
        "/kaggle/working/cache/goemotions_baseline_raw_aligned_fp16_v3.pt"
    )
    assert training_config["training"]["output_dir"].endswith(
        "encoder_eat_comparable_a010_lr1e4"
    )

    generation_keys = (
        "max_eval_examples",
        "batch_size",
        "max_new_tokens",
        "num_beams",
        "do_sample",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "length_penalty",
        "early_stopping",
    )
    for key in generation_keys:
        assert evaluation_config["evaluation"][key] == reference_evaluation_config[
            "evaluation"
        ][key]
    assert evaluation_config["evaluation"]["checkpoint_path"] == training_config[
        "training"
    ]["output_dir"]
    assert scoring_config["scoring"]["input_path"] == evaluation_config["evaluation"][
        "output_path"
    ]
    assert scoring_config["scoring"]["validation_loss_path"] == training_config[
        "training"
    ]["output_dir"]
    assert gpt_oss_config["llm_judge"]["input_path"] == evaluation_config[
        "evaluation"
    ]["output_path"]
    assert gpt_oss_config["llm_judge"]["model"] == "openai/gpt-oss-120b"
    assert gpt_oss_config["llm_judge"]["max_examples"] == 100
    assert qwen_config["llm_judge"]["input_path"] == evaluation_config["evaluation"][
        "output_path"
    ]
    assert qwen_config["llm_judge"]["model"] == "qwen/qwen3.6-27b"
    assert qwen_config["llm_judge"]["max_examples"] == 100
    assert qwen_config["llm_judge"]["reasoning_effort"] == "none"
    assert qwen_config["llm_judge"]["response_format_json"] is True
    assert aggregate_config["judge_aggregation"]["require_all_judges"] is True
    assert aggregate_config["judge_aggregation"]["min_judged_examples"] == 95
    assert aggregate_config["judge_aggregation"]["judges"][0][
        "summary_path"
    ] == gpt_oss_config["llm_judge"]["summary_output_path"]
    assert aggregate_config["judge_aggregation"]["judges"][1][
        "summary_path"
    ] == qwen_config["llm_judge"]["summary_output_path"]


def test_comparable_alpha_005_changes_only_alpha_and_artifact_paths() -> None:
    reference_training = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a010_lr1e4.yaml"
    )
    training_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a005_lr1e4.yaml"
    )
    reference_evaluation = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a010_lr1e4_evaluate.yaml"
    )
    evaluation_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a005_lr1e4_evaluate.yaml"
    )
    reference_scoring = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a010_lr1e4_score.yaml"
    )
    scoring_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a005_lr1e4_score.yaml"
    )
    gpt_oss_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a005_lr1e4_judge_gpt_oss.yaml"
    )
    qwen_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a005_lr1e4_judge_qwen.yaml"
    )
    aggregate_config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a005_lr1e4_judge_aggregate.yaml"
    )

    expected_training = deepcopy(reference_training)
    expected_training["model"]["alpha"] = 0.05
    expected_training["training"]["output_dir"] = (
        "/kaggle/working/models/encoder_eat_comparable_a005_lr1e4"
    )
    expected_training["eat_signal"]["output_path"] = (
        "/kaggle/working/reports/encoder_eat_comparable_a005_lr1e4_r_h.csv"
    )
    assert training_config == expected_training

    expected_evaluation = deepcopy(reference_evaluation)
    expected_evaluation["model"]["alpha"] = 0.05
    expected_evaluation["training"]["output_dir"] = expected_training["training"][
        "output_dir"
    ]
    expected_evaluation["eat_signal"]["output_path"] = expected_training[
        "eat_signal"
    ]["output_path"]
    expected_evaluation["evaluation"]["checkpoint_path"] = expected_training[
        "training"
    ]["output_dir"]
    expected_evaluation["evaluation"]["output_path"] = (
        "/kaggle/working/reports/encoder_eat_comparable_a005_lr1e4_generations.csv"
    )
    assert evaluation_config == expected_evaluation

    expected_scoring = deepcopy(reference_scoring)
    expected_scoring["scoring"]["input_path"] = evaluation_config["evaluation"][
        "output_path"
    ]
    expected_scoring["scoring"]["output_path"] = (
        "/kaggle/working/reports/encoder_eat_comparable_a005_lr1e4_metrics.csv"
    )
    expected_scoring["scoring"]["validation_loss_path"] = expected_training[
        "training"
    ]["output_dir"]
    assert scoring_config == expected_scoring

    assert gpt_oss_config["llm_judge"]["input_path"] == evaluation_config[
        "evaluation"
    ]["output_path"]
    assert gpt_oss_config["llm_judge"]["model"] == "openai/gpt-oss-120b"
    assert gpt_oss_config["llm_judge"]["max_examples"] == 100
    assert qwen_config["llm_judge"]["input_path"] == evaluation_config["evaluation"][
        "output_path"
    ]
    assert qwen_config["llm_judge"]["model"] == "qwen/qwen3.6-27b"
    assert qwen_config["llm_judge"]["max_examples"] == 100
    assert qwen_config["llm_judge"]["reasoning_effort"] == "none"
    assert qwen_config["llm_judge"]["response_format_json"] is True
    assert aggregate_config["judge_aggregation"]["require_all_judges"] is True
    assert aggregate_config["judge_aggregation"]["min_judged_examples"] == 95
    assert aggregate_config["judge_aggregation"]["judges"][0][
        "summary_path"
    ] == gpt_oss_config["llm_judge"]["summary_output_path"]
    assert aggregate_config["judge_aggregation"]["judges"][1][
        "summary_path"
    ] == qwen_config["llm_judge"]["summary_output_path"]


def test_probability_compose_experiment_changes_only_formula_and_artifacts() -> None:
    reference = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a005_lr1e4.yaml"
    )
    training = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_pcompose_a005_lr1e4.yaml"
    )
    evaluation = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_pcompose_a005_lr1e4_evaluate.yaml"
    )
    scoring = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_pcompose_a005_lr1e4_score.yaml"
    )
    gpt_oss = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_pcompose_a005_lr1e4_judge_gpt_oss.yaml"
    )
    qwen = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_pcompose_a005_lr1e4_judge_qwen.yaml"
    )
    aggregate = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_pcompose_a005_lr1e4_judge_aggregate.yaml"
    )

    expected = deepcopy(reference)
    expected["model"]["attention_formula"] = "probability_compose"
    expected["training"]["output_dir"] = (
        "/kaggle/working/models/encoder_eat_comparable_pcompose_a005_lr1e4"
    )
    expected["eat_signal"]["output_path"] = (
        "/kaggle/working/reports/encoder_eat_comparable_pcompose_a005_lr1e4_r_h.csv"
    )
    assert training == expected
    assert evaluation["evaluation"]["checkpoint_path"] == training["training"][
        "output_dir"
    ]
    assert scoring["scoring"]["validation_loss_path"] == training["training"][
        "output_dir"
    ]
    assert gpt_oss["llm_judge"]["max_examples"] == 100
    assert qwen["llm_judge"]["max_examples"] == 100
    assert aggregate["judge_aggregation"]["judges"][0]["summary_path"] == (
        gpt_oss["llm_judge"]["summary_output_path"]
    )
    assert aggregate["judge_aggregation"]["judges"][1]["summary_path"] == (
        qwen["llm_judge"]["summary_output_path"]
    )


def test_tied_orthogonal_experiment_changes_only_initialization_and_artifacts() -> None:
    reference = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a005_lr1e4.yaml"
    )
    training = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_tied_orthogonal_a005_lr1e4.yaml"
    )
    evaluation = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_tied_orthogonal_a005_lr1e4_evaluate.yaml"
    )
    scoring = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_tied_orthogonal_a005_lr1e4_score.yaml"
    )
    diagnostic = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_tied_orthogonal_a005_lr1e4_diagnostic.yaml"
    )
    gpt_oss = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_tied_orthogonal_a005_lr1e4_judge_gpt_oss.yaml"
    )
    qwen = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_tied_orthogonal_a005_lr1e4_judge_qwen.yaml"
    )
    aggregate = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_tied_orthogonal_a005_lr1e4_judge_aggregate.yaml"
    )

    expected = deepcopy(reference)
    expected["model"]["emotion_weight_initialization"] = "tied_orthogonal"
    expected["training"]["output_dir"] = (
        "/kaggle/working/models/encoder_eat_comparable_tied_orthogonal_a005_lr1e4"
    )
    expected["eat_signal"]["output_path"] = (
        "/kaggle/working/reports/"
        "encoder_eat_comparable_tied_orthogonal_a005_lr1e4_r_h.csv"
    )
    assert training == expected
    assert training["training"]["num_train_epochs"] == 40
    assert training["training"]["early_stopping_patience"] == 3
    assert training["model"]["attention_formula"] == "probability_mix"
    assert training["model"]["alpha"] == 0.05
    assert training["training"]["learning_rate"] == 0.0001
    assert evaluation["evaluation"]["checkpoint_path"] == training["training"][
        "output_dir"
    ]
    assert scoring["scoring"]["validation_loss_path"] == training["training"][
        "output_dir"
    ]
    assert diagnostic["diagnostic"]["checkpoint_dir"] == training["training"][
        "output_dir"
    ]
    assert gpt_oss["llm_judge"]["max_examples"] == 100
    assert qwen["llm_judge"]["max_examples"] == 100
    assert aggregate["judge_aggregation"]["judges"][0]["summary_path"] == (
        gpt_oss["llm_judge"]["summary_output_path"]
    )
    assert aggregate["judge_aggregation"]["judges"][1]["summary_path"] == (
        qwen["llm_judge"]["summary_output_path"]
    )


def test_alpha_005_learning_diagnostic_is_bounded_and_uses_training_output() -> None:
    config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a005_lr1e4_diagnostic.yaml"
    )

    assert config["model"]["alpha"] == 0.05
    assert config["training"]["learning_rate"] == 0.0001
    assert config["diagnostic"]["checkpoint_dir"] == config["training"]["output_dir"]
    assert config["diagnostic"]["max_attention_batches"] == 10
    assert config["diagnostic"]["generation_max_examples"] == 50
    assert config["data"]["contextual_emotion_cache"]["path"].startswith(
        "/kaggle/working/"
    )


def test_uploaded_alpha_005_diagnostic_uses_saved_test_artifact() -> None:
    config = load_yaml_config(
        "configs/kaggle_encoder_eat_comparable_a005_lr1e4_diagnostic_uploaded.yaml"
    )

    artifact_root = "/kaggle/input/datasets/cheikhmohamedahid/a005-test"
    assert config["data"]["dataset_path"] == (
        f"{artifact_root}/reports/encoder_eat_comparable_a005_lr1e4_generations.csv"
    )
    assert config["data"]["response_column"] == "reference_response"
    assert config["diagnostic"]["checkpoint_dir"].startswith(artifact_root)
    assert config["data"]["contextual_emotion_cache"]["path"].startswith(
        artifact_root
    )
    assert config["diagnostic"]["dataset_scope"] == "all"
    assert config["diagnostic"]["allow_cache_subset"] is True
