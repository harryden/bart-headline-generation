# BART Headline Generation with LoRA

Fine-tuning experiment for generating news headlines from short article descriptions using `facebook/bart-base` and LoRA adapters.

This was built as a DAT410 Design of AI Systems course project at Chalmers University in Spring 2025 with Elvina Fahlgren. The project received 100/100, and the original course report is available at [`docs/report.pdf`](docs/report.pdf).

## What This Project Shows

The project explores whether parameter-efficient fine-tuning can adapt a pretrained sequence-to-sequence model to headline generation under limited compute.

Key technical pieces:

- Fine-tuning `facebook/bart-base` for description-to-headline generation
- Parameter-efficient training with LoRA instead of updating all model weights
- Hugging Face `transformers`, `datasets`, `evaluate`, and `peft`
- Exploratory analysis of the HuffPost News Category Dataset
- Training and evaluation in Google Colab on an A100 GPU

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
|-- docs/
|   `-- report.pdf
|-- notebooks/
|   |-- 01_final_model.ipynb
|   `-- 02_load_and_explore.ipynb
|-- .gitignore
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

The current notebooks are designed for Google Colab.

Install dependencies in Colab:

```python
!pip install --upgrade "transformers[torch]" datasets evaluate nltk rouge_score
!pip install --upgrade accelerate peft
```

Then:

1. Download `News_Category_Dataset_v3.json` from Kaggle.
2. Upload it to the Google Drive path shown above.
3. Run the exploration notebook.
4. Run the final model notebook on a GPU runtime.

## Current Limitations

This repository currently preserves the course-project version of the work. Before treating it as a fully reproducible ML project, these items should be cleaned up:

- Add a fixed random seed for the train/test split.
- Pin dependencies in `requirements.txt` or `environment.yml`.
- Replace hardcoded Colab paths with configurable paths.
- Add markdown explanations inside the notebooks.
- Resolve the mismatch between saved output at epoch 5 and `num_train_epochs=10` in the notebook.
- Add an inference-only demo path.
- Add a license and clearer dataset usage notes.

## Authorship

Group project by Harry Denell and Elvina Fahlgren for DAT410, Chalmers University.
