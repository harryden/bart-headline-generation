# BART Headline Generation with LoRA

Fine-tuned a small BART model with LoRA to generate headlines from article summaries.

This was built as a DAT410 Design of AI Systems course project at Chalmers University of Technology in Spring 2025 with Elvina Fahlgren. The project received 100/100, and the original course report is available at [`docs/report.pdf`](docs/report.pdf).

## Technical Implementation

The project fine-tunes `facebook/bart-base` for headline generation from short article descriptions using Low-Rank Adaptation (LoRA):

- Target modules: LoRA adapters applied to attention projection layers (`q_proj`, `v_proj`)
- Parameter efficiency: updates 442,368 of 139,862,784 parameters (0.32%)
- Stack: Hugging Face `transformers`, `peft`, `datasets`, and `evaluate`
- Training configuration: batch size 16, fp16 mixed precision on an A100 GPU
- CLI entry points: standalone scripts for training, evaluation, and single-input inference

## Project Snapshot

| Area | Details |
| --- | --- |
| Task | Generate a headline from a short news description |
| Dataset | HuffPost News Category Dataset, about 209k articles |
| Base model | `facebook/bart-base` |
| Adaptation method | LoRA on BART attention projection layers |
| Trainable parameters | 442,368 of 139,862,784, about 0.32% |
| Training environment | Google Colab Pro, A100 GPU, fp16 |
| Batch size | 16 |
| Saved notebook output | Evaluation after epoch 5 |

## Results and Metric Note

The committed notebook reports:

```text
eval_loss: 4.6797
eval_bleu: 0.5041
epoch: 5.0
```

Important: the notebook's `compute_metrics` function multiplies Hugging Face's raw BLEU value by 100 before returning it. That means the reported `eval_bleu: 0.5041` corresponds to raw corpus BLEU of about `0.005`.

This should not be read as `0.50` raw BLEU or 50% BLEU. The result is best understood as a working fine-tuning pipeline built under course constraints, not as a model optimized for production headline quality.

For headline generation, BLEU is also a limited metric: many valid headlines can share little exact n-gram overlap with the reference. A stronger follow-up would add ROUGE, BERTScore, qualitative examples, and a reproducible evaluation script.

## Repository Layout

```text
.
|-- configs/
|   `-- default.json
|-- docs/
|   `-- report.pdf
|-- notebooks/
|   |-- 01_final_model.ipynb
|   `-- 02_load_and_explore.ipynb
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data.py
|   |-- evaluate.py
|   |-- infer.py
|   |-- metrics.py
|   |-- modeling.py
|   `-- train.py
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Notebooks

Run these in order:

1. [`notebooks/02_load_and_explore.ipynb`](notebooks/02_load_and_explore.ipynb) explores the dataset, category distribution, word counts, and missing values.
2. [`notebooks/01_final_model.ipynb`](notebooks/01_final_model.ipynb) loads BART, applies LoRA, preprocesses the dataset, trains, evaluates, and prints sample predictions.

## Dataset

The dataset is not included in this repository.

Use the Kaggle "News Category Dataset" by Rishabh Misra. The notebooks expect the JSON file at:

```text
/content/drive/MyDrive/News_Category_Dataset_v3.json
```

The training notebook also writes checkpoints to:

```text
/content/drive/MyDrive/checkpoints
```

## Running the Project

The original run was done in Google Colab. The repository now also includes script entry points so the workflow can be rerun outside the notebooks.

Install dependencies:

```bash
pip install -r requirements.txt
```

The default config expects the Kaggle dataset at:

```text
/content/drive/MyDrive/News_Category_Dataset_v3.json
```

To use a different location, edit `data.dataset_path` in [`configs/default.json`](configs/default.json).

### Training

```bash
python -m src.train --config configs/default.json
```

This trains LoRA adapters with a fixed train/test split seed and writes checkpoints to:

```text
checkpoints/bart-lora-headline
```

### Evaluation

Evaluate a trained adapter:

```bash
python -m src.evaluate \
  --config configs/default.json \
  --adapter-path checkpoints/bart-lora-headline/final
```

Omit `--adapter-path` to evaluate the base `facebook/bart-base` model as a baseline.

For a quick smoke test on a smaller subset:

```bash
python -m src.evaluate \
  --config configs/default.json \
  --adapter-path checkpoints/bart-lora-headline/final \
  --limit 100
```

The script writes aggregate metrics and sample predictions under `results/`.

### Inference

Generate a headline from one description:

```bash
python -m src.infer \
  --config configs/default.json \
  --adapter-path checkpoints/bart-lora-headline/final \
  --text "A short news article description goes here."
```

Omit `--adapter-path` to generate with the base model.

### Notebook Path

1. Download `News_Category_Dataset_v3.json` from Kaggle.
2. Upload it to the Google Drive path shown above.
3. Run the exploration notebook.
4. Run the final model notebook on a GPU runtime.

## Project Scope and Limitations

This repository preserves the coursework implementation. Evaluation relies on n-gram token overlap (BLEU), which has limited correlation with semantic adequacy in abstractive headline generation. Model training was originally executed on Google Colab, and local script execution requires downloading the external Kaggle dataset.

## Authorship

Group project by Harry Denell and Elvina Fahlgren for DAT410, Chalmers University of Technology.
