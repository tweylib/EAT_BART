from eat_bart.utils.config import load_yaml_config


def test_kaggle_config_inherits_default_config() -> None:
    config = load_yaml_config("configs/kaggle.yaml")

    assert config["model"]["name"] == "facebook/bart-base"
    assert config["data"]["dataset_path"].endswith("finalMentalHealthDataset-question-response.csv")
    assert config["training"]["output_dir"] == "/kaggle/working/models/bart_baseline"
    assert "nrc_lexicon_path" not in config["data"]
    assert not any(key.startswith("modify_") for key in config["model"])


def test_kaggle_baseline_5epoch_matches_comparison_training_protocol() -> None:
    config = load_yaml_config("configs/kaggle_baseline_5epoch.yaml")

    assert config["training"]["num_train_epochs"] == 30
    assert config["training"]["learning_rate"] == 0.00003
    assert config["training"]["gradient_accumulation_steps"] == 8
    assert config["training"]["output_dir"] == "/kaggle/working/models/bart_baseline_30epoch"
    assert config["training"]["load_best_model_at_end"] is True
    assert config["training"]["metric_for_best_model"] == "eval_loss"
    assert config["training"]["greater_is_better"] is False
    assert config["training"]["early_stopping_patience"] == 2
    assert config["training"]["logging_strategy"] == "epoch"
    assert config["data"]["max_target_length"] == 512


def test_kaggle_baseline_experiment_evaluate_uses_full_test_split() -> None:
    config = load_yaml_config("configs/kaggle_baseline_5epoch_experiment_evaluate.yaml")

    assert config["evaluation"]["model_source"] == "checkpoint"
    assert config["evaluation"]["checkpoint_path"] == "/kaggle/working/models/bart_baseline_30epoch"
    assert config["evaluation"]["max_eval_examples"] is None
    assert config["evaluation"]["max_new_tokens"] == 512


def test_kaggle_baseline_experiment_uses_weighted_two_judge_aggregate() -> None:
    config = load_yaml_config(
        "configs/kaggle_baseline_5epoch_experiment_judge_groq_2judge_aggregate.yaml"
    )

    assert config["judge_aggregation"]["min_judged_examples"] == 1
    assert len(config["judge_aggregation"]["judges"]) == 2


def test_both_kaggle_judges_use_the_full_test_generation_set() -> None:
    llama_config = load_yaml_config(
        "configs/kaggle_baseline_5epoch_experiment_judge_groq.yaml"
    )
    gpt_oss_config = load_yaml_config(
        "configs/kaggle_baseline_5epoch_experiment_judge_groq_gpt_oss.yaml"
    )

    assert llama_config["llm_judge"]["max_examples"] is None
    assert gpt_oss_config["llm_judge"]["max_examples"] is None


def test_comparable_baseline_protocol_is_explicit() -> None:
    training_config = load_yaml_config("configs/kaggle_baseline_comparable.yaml")
    evaluation_config = load_yaml_config("configs/kaggle_baseline_comparable_evaluate.yaml")
    judge_config = load_yaml_config("configs/kaggle_baseline_comparable_judge_gpt_oss.yaml")
    qwen_config = load_yaml_config("configs/kaggle_baseline_comparable_judge_qwen.yaml")
    aggregate_config = load_yaml_config(
        "configs/kaggle_baseline_comparable_judge_aggregate.yaml"
    )

    assert training_config["comparison"]["protocol_id"] == "bart_eat_comparable_v1"
    assert training_config["comparison"]["expected_cuda_devices"] == 2
    assert training_config["model"]["add_prefix_space"] is False
    assert training_config["data"]["max_source_length"] == 256
    assert training_config["data"]["max_target_length"] == 512
    assert training_config["training"]["num_train_epochs"] == 30
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
    assert aggregate_config["judge_aggregation"]["require_all_judges"] is True
    assert aggregate_config["judge_aggregation"]["min_judged_examples"] == 95
