"""Evaluate a base model or LoRA adapter on the held-out split.

Run from the repository root:

    python -m src.evaluate --config configs/default.json --adapter-path checkpoints/bart-lora-headline/final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import load_config, resolve_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate headline generation.")
    parser.add_argument("--config", help="Path to a JSON experiment config.")
    parser.add_argument("--adapter-path", help="Path to trained LoRA adapters.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N examples.")
    parser.add_argument("--results-path", help="Where to write aggregate metrics as JSON.")
    parser.add_argument("--samples-path", help="Where to write sample predictions as JSONL.")
    return parser.parse_args()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()

    from src.data import clean_text, load_headline_dataset
    from src.metrics import compute_text_metrics
    from src.modeling import generate_headlines, load_generation_model, load_tokenizer, pick_device

    config = load_config(resolve_config_path(args.config))
    data_config = config["data"]
    evaluation_config = config["evaluation"]

    tokenizer = load_tokenizer(config["model"]["base_model"])
    model = load_generation_model(config, adapter_path=args.adapter_path)
    device = pick_device()

    split = load_headline_dataset(data_config)["test"]
    if args.limit:
        split = split.select(range(min(args.limit, len(split))))

    text_column = data_config.get("text_column", "description")
    target_column = data_config.get("target_column", "headline")
    descriptions = [
        clean_text(value, "Empty description")
        for value in split[text_column]
    ]
    references = [
        clean_text(value, "Empty headline")
        for value in split[target_column]
    ]

    predictions = generate_headlines(
        model=model,
        tokenizer=tokenizer,
        descriptions=descriptions,
        generation_config=config["generation"],
        device=device,
    )
    metrics = compute_text_metrics(predictions, references)
    metrics["num_examples"] = len(predictions)
    metrics["adapter_path"] = args.adapter_path or ""
    metrics["base_model"] = config["model"]["base_model"]

    results_path = args.results_path or evaluation_config["results_path"]
    samples_path = args.samples_path or evaluation_config["sample_predictions_path"]
    num_samples = evaluation_config.get("num_sample_predictions", 25)

    sample_rows = [
        {
            "description": description,
            "prediction": prediction,
            "reference": reference,
        }
        for description, prediction, reference in zip(
            descriptions[:num_samples],
            predictions[:num_samples],
            references[:num_samples],
        )
    ]

    write_json(results_path, metrics)
    write_jsonl(samples_path, sample_rows)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
