# When Does Auxiliary Classification Help Regression in Practice? A Case Study in Arabic Dialectness Modeling

Abdelrahman Sakr, Marwan Torki, and Nagwa El-Makky. Computer and Systems Engineering Department, Alexandria University.

🔍 [**Live demo**](https://huggingface.co/spaces/asakr-ai/aldi-dialectness-demo) ·
🤗 [**Model**](https://huggingface.co/asakr-ai/ALDi-Arabic-Dialectness-MTL) ·
📄 Paper: {put paper link here}

This repository estimates the **Arabic Level of Dialectness (ALDi)** of a sentence: a continuous score from
**0** (Modern Standard Arabic) to **1** (fully dialectal). A shared AraBERTv2-Twitter encoder is trained jointly on
ALDi regression and an **auxiliary classification task** over 7 ALDi bins with **Focal Loss**. The classification
head shapes the encoder during training and is discarded at inference, so the model costs the same to run as a plain
regression model.

## Results

NADI 2024 Subtask 2, blind test set:

| System | RMSE ↓ |
|---|---|
| **Ours (multi-task, 7 bins, Focal Loss)** | **0.1275** |
| ASOS (Nacar et al., 2024) | 0.1403 |
| AlexUNLP-STM (Sakr et al., 2024) | 0.1406 |
| Sentence-ALDi (Keleg et al., 2023) | 0.2178 |

The released model scores **0.10784** RMSE on the official 107-sentence development set, which
`python evaluate.py --data_file data/dev.tsv` reproduces.

## Installation

```bash
git clone https://github.com/Body123/ALDi-Arabic-Dialectness-MTL.git
cd ALDi-Arabic-Dialectness-MTL
pip install -r requirements.txt
```

Tested with Python 3.9 and 3.11, and `transformers==4.39.3`.

## Use the released model

The model downloads automatically from the Hugging Face Hub on first use (about 540 MB). It runs on **CPU** unless a
CUDA GPU is available, at about 0.3 seconds per sentence (Table 4 of the paper).

```bash
# one or more sentences
python predict.py --text "شو هالحكي يا زلمة"

# a whole file: CSV/TSV with a "sentence" column, or .txt with one sentence per line
python predict.py --input sentences.csv --output predictions.csv
```

From Python:

```python
from aldi_mtl import load_model, score

tokenizer, model, device = load_model()            # or load_model("path/to/checkpoint")
print(score(["هذا الكتاب رائع جدا", "شو هالحكي يا زلمة"], tokenizer, model, device))
```

## Reproduce training

### 1. Data

We use the NADI 2024 Subtask 2 data (AOC-ALDi; Keleg et al., 2023): 102,886 training sentences and 107 development
sentences, as TSV files with (at least) a `sentence` and an `ALDi` column. The data is not redistributed here.
Obtain it from the NADI 2024 shared task organizers and place it as:

```
data/train.tsv
data/dev.tsv
```

### 2. Train

```bash
python train.py --config configs/final_system.json --train_file data/train.tsv
```

The script:

1. bins the ALDi scores into 7 classes: class 0 for MSA (ALDi = 0) and 6 equal-width bins for ALDi > 0;
2. runs stratified 5-fold cross-validation on those bins;
3. trains each fold with the loss `α · MSE + (1 − α) · FocalLoss`, with α = 0.73 and γ = 3;
4. evaluates every 750 steps and keeps the checkpoint with the lowest validation RMSE.

Results go to `runs/final_system/`: one `fold_k/` folder of checkpoints per fold, `cv_results.txt`,
`final_summary.txt` (which lists the best checkpoint of each fold), and plots.

The released model is the **best checkpoint of fold 2**. To train only that fold:

```bash
python train.py --config configs/final_system.json --train_file data/train.tsv --folds 2
```

Add `--train_full` to also train one model on all the training data after cross-validation. A GPU is strongly
recommended. Results can differ slightly between hardware because of GPU non-determinism.

### 3. Evaluate

```bash
python evaluate.py --data_file data/dev.tsv --model runs/final_system/fold_2/checkpoint-XXXX
```

Without `--model`, this evaluates the released model. Per-sentence predictions are written to `predictions.csv`.

### Hyperparameters (`configs/final_system.json`)

| Setting | Value |
|---|---|
| Encoder | `aubmindlab/bert-base-arabertv02-twitter` |
| Auxiliary task | 7 bins (MSA + 6 equal-width), Focal Loss γ = 3 |
| Regression loss / weight | MSE, α = 0.73 |
| Optimizer | AdamW, learning rate 4e-6, weight decay 0.01, polynomial schedule, 500 warmup steps |
| Batch size / epochs | 32 / 5 |
| Gradient clipping | max norm 1.0 |
| Head dropout | 0.25 |
| Max sequence length | 256 |
| Cross-validation | stratified 5-fold, seed 42 |

## Demo

The [live demo](https://huggingface.co/spaces/asakr-ai/aldi-dialectness-demo) runs the model on CPU directly in the
browser (ONNX Runtime Web), so it is free and has no usage limits. The model file (about 270 MB) downloads once and is
then cached by the browser.

The same page is in `docs/index.html` and can be hosted for free on GitHub Pages. In the repository settings, open
**Pages**, set **Source** to *Deploy from a branch*, choose branch `main` and folder `/docs`, and save. The demo is then
served at `https://body123.github.io/ALDi-Arabic-Dialectness-MTL/`.

## Repository structure

```
├── aldi_mtl/
│   ├── model.py          # BertForMultiTask (shared encoder + regression + classification heads), FocalLoss
│   ├── data.py           # TSV loading, ALDi binning, dataset
│   ├── training.py       # model config, metrics, dynamic-alpha callback
│   ├── plots.py          # per-fold and cross-validation plots
│   └── inference.py      # load_model / score
├── configs/
│   └── final_system.json # hyperparameters of the paper's final system
├── docs/
│   └── index.html        # in-browser demo (GitHub Pages)
├── train.py              # k-fold training
├── evaluate.py           # RMSE on a labeled TSV
├── predict.py            # predictions for sentences or files
└── requirements.txt
```

## Citation

```bibtex
{put citation here}
```
