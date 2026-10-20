# Hyperparameters

Training configuration of the released checkpoints in `output/`, also used by `train.py`. Inference does not need this file: `infer.py` reads each model's `*_model_config.json`.

---

## Architecture and training (identical for all datasets)

| Setting                      | Value        |
| ---------------------------- | ------------ |
| GNN backbone                 | `gat`        |
| Fusion                       | `bcmg`       |
| Seed                         | `969696`     |
| `hidden_dim`                 | `256`        |
| `num_gnn_layers`             | `3`          |
| `num_heads`                  | `4`          |
| `dropout`                    | `0.1`        |
| `fusion_rank`                | `128`        |
| `fusion_out_dim`             | `256`        |
| Stage-2 Optuna trials        | `3`          |
| Stage-2 max epochs per trial | `30`         |
| Stage-2 final max epochs     | `100`        |
| Stage-2 early-stopping patience | `10`      |

---

## Per-dataset Stage-1 settings

| Dataset          | Stage-1 batch size | Stage-1 epochs | Temperature | Encoder LR factor |
| ---------------- | -----------------: | -------------: | ----------: | ----------------: |
| `ames`           |                128 |             10 |         0.3 |               1.0 |
| `biodegradation` |                 32 |             10 |         0.5 |               1.0 |
| `eye_corrosion`  |                 32 |             10 |         0.3 |               1.0 |
| `eye_irritation` |                 32 |              4 |         0.5 |               1.0 |
| `herg`           |                128 |              3 |         0.5 |               0.1 |
| `sr_p53`         |                 64 |              5 |         0.5 |               1.0 |

---

## Per-dataset Stage-2 settings (selected by Optuna)

| Dataset                                     | Classifier        | Dropout | Learning rate | Weight decay | Batch size |
| ------------------------------------------- | ----------------- | ------: | ------------: | -----------: | ---------: |
| `ames`, `biodegradation`, `herg`            | `[512, 256, 128]` |  0.1889 |      1.347e-4 |     4.024e-5 |         64 |
| `eye_corrosion`, `eye_irritation`, `sr_p53` | `[512, 256, 128]` |  0.2681 |      3.067e-4 |     1.448e-5 |         64 |

---

## Notes

- Stage-2 learning rate, weight decay, batch size, dropout and classifier layout were tuned per dataset; the chosen values are in each `output/<dataset>/*_best_stage2_hyperparameters.json`.
