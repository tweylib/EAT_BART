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
