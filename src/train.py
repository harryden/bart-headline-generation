"""Train BART with LoRA adapters for headline generation.

Run from the repository root:

    python -m src.train --config configs/default.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config, resolve_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BART with LoRA.")
    parser.add_argument("--config", help="Path to a JSON experiment config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

    from src.data import load_headline_dataset, tokenize_dataset
    from src.metrics import build_compute_metrics
    from src.modeling import build_lora_model, load_tokenizer

    config = load_config(resolve_config_path(args.config))
    training_config = config["training"]
    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config["model"]["base_model"])
    model = build_lora_model(config)
    model.print_trainable_parameters()

    dataset = load_headline_dataset(config["data"])
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, config)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        eval_strategy=training_config.get("eval_strategy", "epoch"),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        learning_rate=training_config.get("learning_rate", 2e-5),
        num_train_epochs=training_config.get("num_train_epochs", 5),
        weight_decay=training_config.get("weight_decay", 0.01),
        save_strategy=training_config.get("save_strategy", "epoch"),
        logging_steps=training_config.get("logging_steps", 100),
        predict_with_generate=training_config.get("predict_with_generate", True),
        fp16=training_config.get("fp16", True),
        report_to=training_config.get("report_to", []),
        label_smoothing_factor=training_config.get("label_smoothing_factor", 0.1),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir / "final"))
    trainer.save_metrics("eval", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
