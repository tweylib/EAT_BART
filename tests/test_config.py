from eat_bart.utils.config import load_yaml_config


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
