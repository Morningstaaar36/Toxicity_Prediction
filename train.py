import os
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

from toxicity import SMILESToPyG, MoleculeDataset, ToxicityPredictor
from toxicity.determinism import SEED
from toxicity.dataset import calculate_class_weights
from toxicity.training import (
    train_stage1_contrastive,
    train_stage2_classification_epoch,
    evaluate_stage2,
)


HERE = os.path.dirname(os.path.abspath(__file__))

STAGE1 = {
    'ames':           {'batch_size': 128, 'epochs': 10, 'temperature': 0.3, 'encoder_lr_factor': 1.0},
    'biodegradation': {'batch_size': 32,  'epochs': 10, 'temperature': 0.5, 'encoder_lr_factor': 1.0},
    'eye_corrosion':  {'batch_size': 32,  'epochs': 10, 'temperature': 0.3, 'encoder_lr_factor': 1.0},
    'eye_irritation': {'batch_size': 32,  'epochs': 4,  'temperature': 0.5, 'encoder_lr_factor': 1.0},
    'herg':           {'batch_size': 128, 'epochs': 3,  'temperature': 0.5, 'encoder_lr_factor': 0.1},
    'sr_p53':         {'batch_size': 64,  'epochs': 5,  'temperature': 0.5, 'encoder_lr_factor': 1.0},
}

STAGE2_A = {
    'classifier_hidden_dims': [512, 256, 128],
    'classifier_dropout': 0.18893268591026857,
    'lr': 0.00013471236772361422,
    'weight_decay': 4.0235187891686954e-05,
    'batch_size': 64,
}

STAGE2_B = {
    'classifier_hidden_dims': [512, 256, 128],
    'classifier_dropout': 0.26812923797556293,
    'lr': 0.0003067003409486709,
    'weight_decay': 1.4480209370581667e-05,
    'batch_size': 64,
}

STAGE2 = {
    'ames': STAGE2_A,
    'biodegradation': STAGE2_A,
    'eye_corrosion': STAGE2_B,
    'eye_irritation': STAGE2_B,
    'herg': STAGE2_A,
    'sr_p53': STAGE2_B,
}

FINAL_MAX_EPOCHS = 100

SMILES_COL = 'smiles_standarized'
LABEL_COL  = 'label'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    seed_everything(seed)


def load_split(dataset, split):
    return pd.read_csv(os.path.join(HERE, 'data', 'dataset_deep_pk', dataset, f'{dataset}_{split}.csv'))


def make_dataset(df, converter):
    return MoleculeDataset(
        smiles_list=df[SMILES_COL].tolist(),
        labels=df[LABEL_COL].tolist(),
        converter=converter,
        cache_graphs=False
    )


def make_loader(dataset, batch_size, shuffle, generator):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        follow_batch=['x'],
        generator=generator
    )


def make_stage1_config(dataset):
    return {
        "hidden_dim": 256,
        "num_gnn_layers": 3,
        "num_heads": 4,
        "dropout": 0.1,
        "use_gatv2": True,
        "fusion_rank": 128,
        "fusion_out_dim": 256,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": STAGE1[dataset]['batch_size'],
        "contrastive_epochs": STAGE1[dataset]['epochs'],
    }


def build_model(stage1_config, stage2_config, device):
    return ToxicityPredictor(
        node_feat_dim=44,
        edge_feat_dim=10,
        lap_pe_dim=8,
        hidden_dim=stage1_config["hidden_dim"],
        num_gnn_layers=stage1_config["num_gnn_layers"],
        num_heads=stage1_config["num_heads"],
        dropout=stage1_config["dropout"],
        classifier_dropout=stage2_config["classifier_dropout"],
        fusion_rank=stage1_config["fusion_rank"],
        fusion_out_dim=stage1_config["fusion_out_dim"],
        classifier_hidden_dims=stage2_config["classifier_hidden_dims"],
        num_classes=1,
        freeze_chemberta=True,
        use_gatv2=stage1_config["use_gatv2"]
    ).to(device)


def make_stage2_optimizer(model, stage2_config, encoder_lr_factor):
    enc_params = (list(model.gat_encoder.parameters()) +
                  list(model.proj_g.parameters()) +
                  list(model.proj_l.parameters()) +
                  list(model.fusion.parameters()))
    enc_ids       = {id(p) for p in enc_params}
    enc_trainable = [p for p in enc_params if p.requires_grad]
    head_params   = [p for p in model.parameters()
                     if id(p) not in enc_ids and p.requires_grad]
    _pg = []
    if enc_trainable:
        _pg.append({"params": enc_trainable,
                    "lr": stage2_config["lr"] * encoder_lr_factor})
    if head_params:
        _pg.append({"params": head_params, "lr": stage2_config["lr"]})
    return torch.optim.AdamW(_pg, weight_decay=stage2_config["weight_decay"])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch, return_embeddings_only=False)
        logits = outputs['logits'].view(-1)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        ys = batch.y.view(-1).detach().cpu().numpy()
        for j in range(len(batch.smiles)):
            rows.append({
                'smiles': batch.smiles[j],
                'label': int(ys[j]),
                'pred': int(preds[j].item()),
                'prob': float(probs[j].item()),
                'logit': float(logits[j].item()),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=sorted(STAGE1))
    parser.add_argument('--out', default='runs')
    args = parser.parse_args()
    dataset = args.dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(HERE, args.out, dataset)
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f'{dataset}_saca_tox')

    seed_all(SEED)

    stage1_config = make_stage1_config(dataset)
    stage2_config = dict(STAGE2[dataset])
    stage2_config["max_epochs"] = FINAL_MAX_EPOCHS

    train_df = load_split(dataset, 'train')
    val_df = load_split(dataset, 'val')
    test_df = load_split(dataset, 'test')
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    converter = SMILESToPyG(pos_enc_dim=8, use_chirality=True, explicit_H=False)
    train_val_dataset = make_dataset(train_val_df, converter)
    test_dataset = make_dataset(test_df, converter)

    class_weights = calculate_class_weights(train_val_df[LABEL_COL].tolist())

    g_final = torch.Generator()
    g_final.manual_seed(SEED)
    train_val_loader_s1 = make_loader(train_val_dataset, stage1_config["batch_size"], True, g_final)
    train_val_loader_s2 = make_loader(train_val_dataset, stage2_config["batch_size"], True, g_final)
    test_loader = make_loader(test_dataset, stage2_config["batch_size"], False, g_final)

    model = build_model(stage1_config, stage2_config, device)

    optimizer_s1 = torch.optim.AdamW(
        model.parameters(),
        lr=stage1_config["lr"],
        weight_decay=stage1_config["weight_decay"]
    )
    start_time_s1 = time.time()
    train_stage1_contrastive(
        model, train_val_loader_s1, optimizer_s1, device,
        num_epochs=stage1_config["contrastive_epochs"],
        temperature=STAGE1[dataset]['temperature']
    )
    training_time_s1 = time.time() - start_time_s1

    pos_weight = class_weights[1] / class_weights[0]
    criterion_s2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer_s2 = make_stage2_optimizer(model, stage2_config, STAGE1[dataset]['encoder_lr_factor'])
    scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s2,
        T_max=stage2_config["max_epochs"],
        eta_min=1e-6
    )

    print("\n" + "="*80)
    print("STAGE 2: CLASSIFICATION FINE-TUNING (TRAIN+VAL)")
    print("="*80)

    start_time_s2 = time.time()
    for epoch in range(1, stage2_config["max_epochs"] + 1):
        train_metrics = train_stage2_classification_epoch(
            model, train_val_loader_s2, criterion_s2, optimizer_s2, device
        )
        scheduler_s2.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{stage2_config['max_epochs']} | Loss: {train_metrics['loss']:.4f} | "
                  f"Acc: {train_metrics['accuracy']:.4f} | Bal_Acc: {train_metrics['balanced_accuracy']:.4f}")
    training_time_s2 = time.time() - start_time_s2

    test_metrics = evaluate_stage2(model, test_loader, criterion_s2, device)

    print(f"\n{'='*80}")
    print(f"FINAL TEST RESULTS  (dataset={dataset})")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    print(f"{'Accuracy':<25} {test_metrics['accuracy']:<15.4f}")
    print(f"{'Balanced Accuracy':<25} {test_metrics['balanced_accuracy']:<15.4f}")
    print(f"{'AUROC':<25} {test_metrics['roc_auc']:<15.4f}")
    print(f"{'AUPRC':<25} {test_metrics['pr_auc']:<15.4f}")
    print(f"{'MCC':<25} {test_metrics['mcc']:<15.4f}")
    print(f"{'F1 Score':<25} {test_metrics['f1']:<15.4f}")
    print(f"{'Precision':<25} {test_metrics['precision']:<15.4f}")
    print(f"{'Recall':<25} {test_metrics['recall']:<15.4f}")
    print(f"{'Specificity':<25} {test_metrics['specificity']:<15.4f}")
    if 'confusion_matrix' in test_metrics:
        cm = test_metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm['tn']:<6} FP: {cm['fp']:<6}")
        print(f"  FN: {cm['fn']:<6} TP: {cm['tp']:<6}")

    torch.save(model.state_dict(), f'{prefix}_model.pt')

    model_config = {
        'dataset'         : dataset,
        'gnn_backbone'    : 'gat',
        'fusion_type'     : 'bcmg',
        'seed'            : SEED,
        'node_feat_dim'   : 44,
        'edge_feat_dim'   : 10,
        'lap_pe_dim'      : 8,
        'hidden_dim'      : stage1_config['hidden_dim'],
        'num_gnn_layers'  : stage1_config['num_gnn_layers'],
        'num_heads'       : stage1_config['num_heads'],
        'dropout'         : stage1_config['dropout'],
        'fusion_rank'     : stage1_config['fusion_rank'],
        'fusion_out_dim'  : stage1_config['fusion_out_dim'],
        'classifier_hidden_dims': stage2_config['classifier_hidden_dims'],
        'num_classes'     : 1,
        'freeze_chemberta': True,
        'use_gatv2'       : stage1_config['use_gatv2'],
        'chemberta_model' : 'seyonec/ChemBERTa-zinc-base-v1',
        'converter'       : {'pos_enc_dim': 8, 'use_chirality': True, 'explicit_H': False},
    }
    with open(f'{prefix}_model_config.json', 'w') as f:
        json.dump(model_config, f, indent=4)

    results = {
        'dataset': dataset,
        'stage1_time_seconds': training_time_s1,
        'stage2_time_seconds': training_time_s2,
        'stage1_hyperparameters': stage1_config,
        'stage2_hyperparameters': stage2_config,
        'test_metrics': {
            'accuracy': test_metrics['accuracy'],
            'balanced_accuracy': test_metrics['balanced_accuracy'],
            'auroc': test_metrics['roc_auc'],
            'auprc': test_metrics['pr_auc'],
            'mcc': test_metrics['mcc'],
            'f1': test_metrics['f1'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'specificity': test_metrics['specificity'],
        },
    }
    with open(f'{prefix}_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    predict(model, test_loader, device).to_csv(f'{prefix}_test_predictions.csv', index=False)
    print(f"\nwrote {out_dir}")


if __name__ == '__main__':
    main()
