from eat_bart.utils.config import load_yaml_config


def test_kaggle_config_inherits_default_config() -> None:
    config = load_yaml_config("configs/kaggle.yaml")

    assert config["model"]["name"] == "facebook/bart-base"
    assert config["data"]["dataset_path"].endswith("finalMentalHealthDataset-question-response.csv")
    assert config["training"]["output_dir"] == "/kaggle/working/models/eat_bart"


def test_kaggle_evaluate_config_inherits_kaggle_config() -> None:
    config = load_yaml_config("configs/kaggle_evaluate.yaml")

    assert config["data"]["dataset_path"].endswith("finalMentalHealthDataset-question-response.csv")
    assert config["evaluation"]["checkpoint_path"] == "/kaggle/working/models/eat_bart"
    assert config["evaluation"]["output_path"] == "/kaggle/working/reports/eat_bart_generations.csv"
