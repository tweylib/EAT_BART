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

    assert config["training"]["num_train_epochs"] == 50
    assert config["training"]["learning_rate"] == 0.00003
    assert config["training"]["gradient_accumulation_steps"] == 8
    assert config["training"]["output_dir"] == "/kaggle/working/models/bart_baseline_50epoch"
    assert config["training"]["load_best_model_at_end"] is True
    assert config["training"]["metric_for_best_model"] == "eval_loss"
    assert config["training"]["greater_is_better"] is False
    assert config["training"]["early_stopping_patience"] == 2
    assert config["training"]["logging_strategy"] == "epoch"
    assert config["data"]["max_target_length"] == 512


def test_kaggle_baseline_experiment_evaluate_uses_full_test_split() -> None:
    config = load_yaml_config("configs/kaggle_baseline_5epoch_experiment_evaluate.yaml")

    assert config["evaluation"]["model_source"] == "checkpoint"
    assert config["evaluation"]["checkpoint_path"] == "/kaggle/working/models/bart_baseline_50epoch"
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
