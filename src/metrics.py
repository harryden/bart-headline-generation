"""Evaluation metrics for generated headlines."""

from __future__ import annotations

from typing import Any

import evaluate
import numpy as np


def decode_token_ids(tokenizer: Any, token_ids: Any) -> list[str]:
    """Decode generated token ids into strings."""
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def compute_text_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute BLEU and ROUGE for headline generation."""
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    bleu_result = bleu_metric.compute(
        predictions=predictions,
        references=[[reference] for reference in references],
    )
    rouge_result = rouge_metric.compute(
        predictions=predictions,
        references=references,
    )

    raw_bleu = float(bleu_result["bleu"])
    return {
        "bleu": raw_bleu,
        "bleu_percent": raw_bleu * 100,
        "rouge1": float(rouge_result["rouge1"]),
        "rouge2": float(rouge_result["rouge2"]),
        "rougeL": float(rouge_result["rougeL"]),
        "rougeLsum": float(rouge_result["rougeLsum"]),
    }


def build_compute_metrics(tokenizer: Any):
    """Build the metric callback used by Seq2SeqTrainer."""
    def compute_metrics(eval_pred: tuple[Any, Any]) -> dict[str, float]:
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_predictions = decode_token_ids(tokenizer, predictions)
        decoded_labels = decode_token_ids(tokenizer, labels)

        return compute_text_metrics(decoded_predictions, decoded_labels)

    return compute_metrics
