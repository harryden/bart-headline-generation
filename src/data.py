"""Dataset loading and preprocessing helpers."""

from __future__ import annotations

from typing import Any

from datasets import DatasetDict, load_dataset


def load_headline_dataset(data_config: dict[str, Any]) -> DatasetDict:
    """Load the HuffPost JSON dataset and create a reproducible split."""
    dataset_path = data_config["dataset_path"]
    text_column = data_config.get("text_column", "description")
    fallback_text_column = data_config.get("fallback_text_column", "short_description")
    target_column = data_config.get("target_column", "headline")

    dataset = load_dataset("json", data_files={"train": dataset_path}, field=None)
    train_dataset = dataset["train"]

    if fallback_text_column in train_dataset.column_names and text_column not in train_dataset.column_names:
        train_dataset = train_dataset.rename_column(fallback_text_column, text_column)

    missing_columns = [
        column
        for column in (text_column, target_column)
        if column not in train_dataset.column_names
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    return train_dataset.train_test_split(
        test_size=data_config.get("test_size", 0.1),
        seed=data_config.get("seed", 42),
    )


def clean_text(value: Any, fallback: str) -> str:
    """Return a non-empty string for tokenizer input."""
    if value is None:
        return fallback

    text = str(value).strip()
    return text if text else fallback


def tokenize_dataset(dataset: DatasetDict, tokenizer: Any, config: dict[str, Any]) -> DatasetDict:
    """Tokenize descriptions and headlines for seq2seq training."""
    data_config = config["data"]
    model_config = config["model"]
    text_column = data_config.get("text_column", "description")
    target_column = data_config.get("target_column", "headline")
    max_input_length = model_config.get("max_input_length", 250)
    max_target_length = model_config.get("max_target_length", 30)

    def preprocess_function(examples: dict[str, list[Any]]) -> dict[str, Any]:
        inputs = [
            clean_text(description, "Empty description")
            for description in examples[text_column]
        ]
        targets = [
            clean_text(headline, "Empty headline")
            for headline in examples[target_column]
        ]

        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
