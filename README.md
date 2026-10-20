# SACA-Tox — Toxicity Prediction

SACA-Tox is a two-stage, contrastively-pretrained molecular property model that fuses a
**graph neural network** (over the molecular graph) with a frozen **ChemBERTa** language
model (over the SMILES string) through a gated bilinear fusion block, and predicts binary
toxicity endpoints.

It is trained and evaluated on six DeepPK benchmark endpoints:

| Dataset (`DATASET` flag) | Endpoint                     | Train |  Val | Test |
| ------------------------ | ---------------------------- | ----: | ---: | ---: |
| `ames`                   | Ames mutagenicity            |  6121 |  766 |  766 |
| `biodegradation`         | Ready biodegradability       |  1329 |  166 |  165 |
| `eye_corrosion`          | Eye corrosion                |  1837 |  231 |  230 |
| `eye_irritation`         | Eye irritation               |  4166 |  521 |  522 |
| `herg`                   | hERG cardiotoxicity          | 16582 | 2073 | 2073 |
| `sr_p53`                 | Tox21 SR-p53 stress response |  5971 |  749 |  746 |

---

## Results (held-out test set)

Configuration: `gat` backbone, `bcmg` fusion. Produced by the **Kaggle**
training runs whose artifacts live in `output/`.

| Dataset        |  AUROC |  AUPRC | Accuracy | Bal. Acc. |    MCC |     F1 |
| -------------- | -----: | -----: | -------: | --------: | -----: | -----: |
| ames           | 0.9128 | 0.9332 |   0.8407 |    0.8343 | 0.6717 | 0.8644 |
| biodegradation | 0.9243 | 0.8428 |   0.8727 |    0.8656 | 0.7143 | 0.8073 |
| eye_corrosion  | 0.9856 | 0.9674 |   0.9565 |    0.9558 | 0.9073 | 0.9419 |
| eye_irritation | 0.9709 | 0.9842 |   0.9559 |    0.9413 | 0.8848 | 0.9703 |
| herg           | 0.9243 | 0.9452 |   0.8601 |    0.8504 | 0.7048 | 0.8863 |
| sr_p53         | 0.8409 | 0.3682 |   0.8847 |    0.7092 | 0.3563 | 0.4110 |

The full per-dataset metrics are in each `output/<dataset>/*_results.json`.

---

## Get the trained models / outputs first

The `output/` folder (trained model weights, predictions, metrics, configs) is **not shipped
in this repository** because each model checkpoint is ~184 MB. Download it from:

**https://drive.google.com/drive/folders/1DOutDgkBe-63RAXAMkdRJegKUtAJfvHt?usp=drive_link**

In that Drive folder you will find **`output.zip`**. Unzip it at the project root so the
layout becomes:

```
Toxicity_Prediction/
├── output/
│   ├── ames/
│   ├── bio_deg/
│   ├── eye_corrosion/
│   ├── eye_irritation/
│   ├── herg/
│   └── sr_p53/
├── data/
├── toxicity/
├── infer.py
└── ...
```

```bash
unzip output.zip      # run from the Toxicity_Prediction/ root
```

Each `output/<dataset>/` directory contains:

| File                                 | What it is                                                |
| ------------------------------------ | --------------------------------------------------------- |
| `*_model.pt`                         | trained model weights (state dict)                        |
| `*_model_config.json`                | architecture config used to rebuild the model             |
| `*_best_stage2_hyperparameters.json` | Optuna-tuned Stage-2 hyperparameters                      |
| `*_results.json` / `*_results.csv`   | final test metrics                                        |
| `*_test_predictions.csv`             | per-molecule predictions from the Kaggle run              |
| `*_standalone_inference.csv`         | per-molecule predictions reproduced locally by `infer.py` |
| `__huggingface_repos__.json`         | (provenance) HF repos the run pushed to                   |

---

## Repository layout

```
train.py          Training. Stage 1 SACA contrastive pretraining, Stage 2 fine-tuning,
                  test evaluation; writes artifacts to runs/<dataset>/.
                  CLI: `python train.py <dataset>`.
infer.py          Local inference. Rebuilds a model from a saved config, loads the checkpoint,
                  runs the test split, writes predictions and metrics.
                  CLI: `python infer.py <dataset>`.
toxicity/         Model code: SMILES-to-graph featurizer, dataset, ChemBERTa wrapper, GAT
                  encoder, BCMG fusion, predictor, SACA losses, training loops, metrics.
data/             DeepPK splits: data/dataset_deep_pk/<dataset>/<dataset>_{train,val,test}.csv
                  Columns: smiles_standarized, label, group
output/           Trained artifacts — download from the Drive link above (see section).
environment.yml   Full, pinned conda export of the `toxicity` env (this machine, Linux/CUDA).
HYPERPARAMETERS.md   Training configuration of the released checkpoints.
```

---

## Environment setup

Create the conda environment (Python 3.10, PyTorch 2.5.1 + CUDA 11.8):

```bash
conda env create -f environment.yml
conda activate toxicity
```

Key package versions (from the `toxicity` env):

| Package         | Version           |
| --------------- | ----------------- |
| python          | 3.10.20           |
| torch           | 2.5.1 (cuda 11.8) |
| torch_geometric | 2.8.0             |
| rdkit           | 2026.3.3          |
| transformers    | 4.44.2            |
| tokenizers      | 0.19.1            |
| huggingface_hub | 0.36.2            |
| scikit-learn    | 1.7.2             |
| pandas          | 2.3.3             |
| numpy           | 1.26.4            |

The first run downloads the ChemBERTa weights `seyonec/ChemBERTa-zinc-base-v1` from the
Hugging Face Hub, so the machine needs internet access on the first inference call.

---

## Reproduce the results (inference only)

No training, just load the released checkpoints and
re-score the test split.

1. Set up the env (above) and unzip `output.zip` (above).
2. Run, for any dataset:

```bash
conda activate toxicity
python infer.py ames
# also: biodegradation | eye_corrosion | eye_irritation | herg | sr_p53
```

`infer.py` will:

- locate the newest `output/<dataset>/*_model_config.json` + `*_model.pt`,
- rebuild the model and load the weights,
- run `data/dataset_deep_pk/<dataset>/<dataset>_test.csv`,
- print the full metric table (AUROC, AUPRC, MCC, F1, …, confusion matrix),
- write `output/<dataset>/<dataset>_standalone_inference.csv`,
- and compare against the Kaggle `*_test_predictions.csv` (prints prediction agreement and
  max prob/logit difference to stderr).

> **Expected drift vs. Kaggle:** local metrics will not match the Kaggle `*_results.json`
> exactly. The Laplacian positional encoding uses eigenvectors whose sign is arbitrary, so PE
> features can flip sign between environments and perturb borderline predictions. The effect
> is small on large/clean splits and more visible on small ones — e.g. `biodegradation`
> (test=165) reproduces at ~0.95 prediction agreement and AUROC ≈ 0.90 vs. 0.9243 on Kaggle.
> The trends and ranking across datasets are preserved; treat the Kaggle `*_results.json` as
> the canonical numbers and `infer.py` as a reproducibility check.

---

## Train from scratch

`train.py` retrains SACA-Tox on one dataset with the per-dataset settings in
`HYPERPARAMETERS.md`: Stage 1 SACA contrastive pretraining, then Stage 2 fine-tuning with the
Stage-2 hyperparameters selected by Optuna for the released checkpoints, both on train+val,
followed by evaluation on the test split.

```bash
conda activate toxicity
python train.py biodegradation
# also: ames | eye_corrosion | eye_irritation | herg | sr_p53
```

Artifacts are written to `runs/<dataset>/` (model weights, model config, best Stage-2
hyperparameters, test metrics, test predictions), so the released checkpoints in `output/`
are never overwritten. A GPU is strongly recommended. Retrained metrics differ from the
released ones, most visibly on the small `biodegradation` test set, for the reason given in
the drift note above and because GPU kernels differ between machines.

---

## Model architecture (summary)

- **Graph encoder:** GAT (3 layers, hidden 256, 4 heads) over RDKit molecular graphs
  (44-dim node features, 10-dim edge features) augmented with an 8-dim Laplacian positional
  encoding. Global readout = multi-head attention + mean/max pooling.
- **Language encoder:** frozen `ChemBERTa-zinc-base-v1` over the SMILES string.
- **Fusion:** `bcmg` gated bilinear fusion (rank 128 → 256-dim joint embedding).
- **Stage 1:** SACA contrastive pretraining of the joint embedding.
- **Stage 2:** MLP classifier `[512, 256, 128] → 1`, BCE-with-logits + positive class
  weighting, hyperparameters tuned by Optuna per dataset.
- **Seed:** 969696 (deterministic flags set in `toxicity/determinism.py`).
