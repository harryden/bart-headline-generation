# BART Headline Generation

Fine-tuned BART to generate news headlines from article descriptions. Used LoRA for parameter-efficient training.

## Overview

This project fine-tunes facebook/bart-base on 209k HuffPost articles to generate headlines from short article descriptions. Instead of training all 139M parameters, LoRA lets you train adapter layers while keeping the base model frozen.

**Key Results:**
* BLEU score: ~0.03 at epoch 1 → ~0.50 after 5 epochs
* Trainable parameters: 442k out of 139M (0.32%)
* Training: Google Colab Pro, A100 GPU, batch size 16, fp16, 5 epochs

**LoRA Config:**
* r=8, lora_alpha=32
* Target modules: q_proj, v_proj

## Dataset

HuffPost News Category Dataset (~209k articles). Not included in this repo.

Available on Kaggle: "News Category Dataset" by Rishabh Misra.

## Usage

Designed for Google Colab. You'll need the dataset in your Google Drive at:
```
/content/drive/MyDrive/News_Category_Dataset_v3.json
```

**Notebooks:**
1. `02_load_and_explore.ipynb` — Dataset exploration and analysis
2. `01_final_model.ipynb` — BART fine-tuning with LoRA, training loop, evaluation

Run the exploration notebook first to understand the data, then the model notebook to train.

## Background

Created for DAT410 (Design of AI Systems) at Chalmers University, Spring 2025. Group project with Elvina Fahlgren. Scored 100/100.
