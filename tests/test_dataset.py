from pathlib import Path

from eat_bart.data.dataset import (
    MentalHealthResponseDataset,
    QuestionResponseExample,
    load_question_response_csv,
    split_dataset,
)


def test_load_question_response_csv_ignores_index_column(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dataset.csv"
    dataset_file.write_text(
        ',question,response\n0,"What helps?","Breathing can help."\n',
        encoding="utf-8",
    )

    examples = load_question_response_csv(dataset_file)

    assert len(examples) == 1
    assert examples[0].question == "What helps?"
    assert examples[0].response == "Breathing can help."


def test_mental_health_response_dataset_returns_question_response(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dataset.csv"
    dataset_file.write_text(
        ',question,response\n0,"What helps?","Breathing can help."\n',
        encoding="utf-8",
    )

    dataset = MentalHealthResponseDataset.from_csv(dataset_file)

    assert len(dataset) == 1
    assert dataset[0] == {"question": "What helps?", "response": "Breathing can help."}


def test_load_question_response_csv_cleans_html_artifacts(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dataset.csv"
    dataset_file.write_text(
        (
            "question,response\n"
            '"Am I overreacting?",'
            '"< p > I&apos;m sorry to hear about this . &nbsp; < / p > '
            '< p style = "" margin - bottom : 0 in 8 pt ; font - size : 11.5 pt ; "" > '
            '< br > It sounds painful. < / p >"\n'
        ),
        encoding="utf-8",
    )

    examples = load_question_response_csv(dataset_file)

    assert examples[0].response == "I'm sorry to hear about this. It sounds painful."
    assert "<" not in examples[0].response
    assert "&nbsp;" not in examples[0].response
    assert "font-size" not in examples[0].response


def test_split_dataset_is_deterministic() -> None:
    dataset = MentalHealthResponseDataset(
        examples=[
            QuestionResponseExample(question=str(index), response=str(index))
            for index in range(10)
        ]
    )

    train_a, validation_a, test_a = split_dataset(dataset, validation_size=0.2, test_size=0.2, seed=7)
    train_b, validation_b, test_b = split_dataset(dataset, validation_size=0.2, test_size=0.2, seed=7)

    assert train_a.indices == train_b.indices
    assert validation_a.indices == validation_b.indices
    assert test_a.indices == test_b.indices
    assert len(train_a) == 6
    assert len(validation_a) == 2
    assert len(test_a) == 2
