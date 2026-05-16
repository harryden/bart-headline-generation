"""Model construction and generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_tokenizer(model_name: str):
    """Load the tokenizer for the configured base model."""
    return AutoTokenizer.from_pretrained(model_name)


def build_lora_model(config: dict[str, Any]):
    """Load the base seq2seq model and attach LoRA adapters."""
    model_config = config["model"]
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_config["base_model"])
    lora_config = model_config["lora"]

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
    )
    return get_peft_model(base_model, peft_config)


def load_generation_model(config: dict[str, Any], adapter_path: str | None = None):
    """Load either the base model or a base model with trained LoRA adapters."""
    model_name = config["model"]["base_model"]
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if adapter_path:
        return PeftModel.from_pretrained(base_model, str(Path(adapter_path)))

    return base_model


def pick_device() -> torch.device:
    """Choose the best available local device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_headlines(
    model: Any,
    tokenizer: Any,
    descriptions: list[str],
    generation_config: dict[str, Any],
    device: torch.device | None = None,
) -> list[str]:
    """Generate one headline for each description."""
    active_device = device or pick_device()
    model.to(active_device)
    model.eval()
    batch_size = generation_config.get("batch_size", 16)
    predictions: list[str] = []

    for start_index in range(0, len(descriptions), batch_size):
        batch_descriptions = descriptions[start_index:start_index + batch_size]
        encoded_inputs = tokenizer(
            batch_descriptions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded_inputs = {
            key: value.to(active_device)
            for key, value in encoded_inputs.items()
        }

        with torch.no_grad():
            output_ids = model.generate(
                **encoded_inputs,
                max_new_tokens=generation_config.get("max_new_tokens", 30),
                num_beams=generation_config.get("num_beams", 4),
                early_stopping=generation_config.get("early_stopping", True),
            )

        predictions.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))

    return predictions
