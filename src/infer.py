"""Generate headlines from the command line.

Run from the repository root:

    python -m src.infer --text "A short news article description..."
"""

from __future__ import annotations

import argparse

from src.config import load_config, resolve_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a headline.")
    parser.add_argument("--config", help="Path to a JSON experiment config.")
    parser.add_argument("--adapter-path", help="Path to trained LoRA adapters.")
    parser.add_argument("--text", required=True, help="Article description to summarize as a headline.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.modeling import generate_headlines, load_generation_model, load_tokenizer, pick_device

    config = load_config(resolve_config_path(args.config))

    tokenizer = load_tokenizer(config["model"]["base_model"])
    model = load_generation_model(config, adapter_path=args.adapter_path)
    predictions = generate_headlines(
        model=model,
        tokenizer=tokenizer,
        descriptions=[args.text],
        generation_config=config["generation"],
        device=pick_device(),
    )
    print(predictions[0])


if __name__ == "__main__":
    main()
