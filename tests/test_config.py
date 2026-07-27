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

    assert config["training"]["num_train_epochs"] == 5
    assert config["training"]["learning_rate"] == 0.00003
    assert config["training"]["gradient_accumulation_steps"] == 8
    assert config["training"]["output_dir"] == "/kaggle/working/models/bart_baseline_5epoch"


def test_kaggle_baseline_experiment_evaluate_uses_full_test_split() -> None:
    config = load_yaml_config("configs/kaggle_baseline_5epoch_experiment_evaluate.yaml")

    assert config["evaluation"]["model_source"] == "checkpoint"
    assert config["evaluation"]["checkpoint_path"] == "/kaggle/working/models/bart_baseline_5epoch"
    assert config["evaluation"]["max_eval_examples"] is None


def test_kaggle_baseline_experiment_uses_weighted_two_judge_aggregate() -> None:
    config = load_yaml_config(
        "configs/kaggle_baseline_5epoch_experiment_judge_groq_2judge_aggregate.yaml"
    )

    assert config["judge_aggregation"]["min_judged_examples"] == 1
    assert len(config["judge_aggregation"]["judges"]) == 2
