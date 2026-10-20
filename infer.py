import os
import sys
import glob
import json
import argparse
import contextlib
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from toxicity import SMILESToPyG, MoleculeDataset, ToxicityPredictor, compute_metrics


HERE = os.path.dirname(os.path.abspath(__file__))


def find_artifacts(dataset):
    cfgs = glob.glob(os.path.join(HERE, 'output', '*', f'{dataset}_*_model_config.json'))
    if not cfgs:
        raise FileNotFoundError(f'no model_config.json for dataset {dataset} under output/')
    config_path = max(cfgs, key=os.path.getmtime)
    artifact_dir = os.path.dirname(config_path)
    pts = glob.glob(os.path.join(artifact_dir, f'{dataset}_*_model.pt'))
    if not pts:
        raise FileNotFoundError(f'no *_model.pt for dataset {dataset} in {artifact_dir}')
    return config_path, max(pts, key=os.path.getmtime), artifact_dir


def build_model(cfg, device):
    if cfg['gnn_backbone'] != 'gat' or cfg['fusion_type'] != 'bcmg':
        raise ValueError('unsupported checkpoint: expected gat backbone with bcmg fusion')
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        model = ToxicityPredictor(
            node_feat_dim=cfg['node_feat_dim'],
            edge_feat_dim=cfg['edge_feat_dim'],
            lap_pe_dim=cfg['lap_pe_dim'],
            hidden_dim=cfg['hidden_dim'],
            num_gnn_layers=cfg['num_gnn_layers'],
            num_heads=cfg['num_heads'],
            dropout=cfg['dropout'],
            fusion_rank=cfg['fusion_rank'],
            fusion_out_dim=cfg['fusion_out_dim'],
            classifier_hidden_dims=cfg['classifier_hidden_dims'],
            num_classes=cfg['num_classes'],
            freeze_chemberta=cfg['freeze_chemberta'],
            use_gatv2=cfg['use_gatv2'],
            chemberta_model=cfg['chemberta_model'],
        )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()
    dataset = args.dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path, pt_path, artifact_dir = find_artifacts(dataset)

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    model = build_model(cfg, device)
    state = torch.load(pt_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state and not any(k.startswith('gat_encoder') for k in state):
        state = state['state_dict']
    incompat = model.load_state_dict(state, strict=False)
    real_missing = [k for k in incompat.missing_keys if not k.startswith('chemberta.')]
    real_unexpected = [k for k in incompat.unexpected_keys if not k.startswith('chemberta.')]
    if real_missing or real_unexpected:
        sys.stderr.write(f'WARNING missing={real_missing[:5]} unexpected={real_unexpected[:5]}\n')
    model.eval()

    test_csv = os.path.join(HERE, 'data', 'dataset_deep_pk', dataset, f'{dataset}_test.csv')
    test_df = pd.read_csv(test_csv)
    smiles_col = 'smiles_standarized' if 'smiles_standarized' in test_df.columns else 'smiles'
    label_col = 'label'

    conv_cfg = cfg.get('converter', {'pos_enc_dim': cfg['lap_pe_dim'], 'use_chirality': True, 'explicit_H': False})
    converter = SMILESToPyG(
        pos_enc_dim=conv_cfg['pos_enc_dim'],
        use_chirality=conv_cfg['use_chirality'],
        explicit_H=conv_cfg['explicit_H'],
    )
    test_dataset = MoleculeDataset(
        smiles_list=test_df[smiles_col].tolist(),
        labels=test_df[label_col].tolist(),
        converter=converter,
        cache_graphs=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        follow_batch=['x'],
    )

    has_gate = hasattr(model.fusion, 'gate_graph_fraction')
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch, return_embeddings_only=False)
            logits = outputs['logits'].view(-1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            gate = outputs.get('gate', None)
            if has_gate and gate is not None:
                gfrac = model.fusion.gate_graph_fraction(gate).detach().cpu().numpy()
            else:
                gfrac = np.full(logits.shape[0], np.nan)
            smis = batch.smiles
            ys = batch.y.view(-1).detach().cpu().numpy()
            for j in range(len(smis)):
                rows.append({
                    'smiles': smis[j],
                    'label': int(ys[j]),
                    'pred': int(preds[j].item()),
                    'prob': float(probs[j].item()),
                    'logit': float(logits[j].item()),
                    'gate_graph_frac': float(gfrac[j]),
                })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(artifact_dir, f'{dataset}_standalone_inference.csv')
    out_df.to_csv(out_path, index=False)
    sys.stderr.write(f'wrote {out_path}\n')

    y_true = out_df['label'].values
    y_pred = out_df['pred'].values
    y_scores = out_df['prob'].values
    m = compute_metrics(y_true, y_pred, y_scores)
    print(f"\n{'='*80}")
    print(f"FINAL TEST RESULTS  (dataset={dataset})")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    print(f"{'Accuracy':<25} {m['accuracy']:<15.4f}")
    print(f"{'Balanced Accuracy':<25} {m['balanced_accuracy']:<15.4f}")
    print(f"{'AUROC':<25} {m['roc_auc']:<15.4f}")
    print(f"{'AUPRC':<25} {m['pr_auc']:<15.4f}")
    print(f"{'MCC':<25} {m['mcc']:<15.4f}")
    print(f"{'F1 Score':<25} {m['f1']:<15.4f}")
    print(f"{'Precision':<25} {m['precision']:<15.4f}")
    print(f"{'Recall':<25} {m['recall']:<15.4f}")
    print(f"{'Specificity':<25} {m['specificity']:<15.4f}")
    if 'confusion_matrix' in m:
        cm = m['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm['tn']:<6} FP: {cm['fp']:<6}")
        print(f"  FN: {cm['fn']:<6} TP: {cm['tp']:<6}")

    kag_matches = sorted(glob.glob(os.path.join(artifact_dir, f'{dataset}_*_test_predictions.csv')))
    if kag_matches:
        kag = pd.read_csv(kag_matches[0])
        n = min(len(kag), len(out_df))
        a = out_df.iloc[:n].reset_index(drop=True)
        k = kag.iloc[:n].reset_index(drop=True)
        pred_agree = float((a['pred'].values == k['pred'].values).mean())
        prob_mad = float(np.abs(a['prob'].values - k['prob'].values).max())
        logit_mad = float(np.abs(a['logit'].values - k['logit'].values).max())
        same_smiles = bool((a['smiles'].values == k['smiles'].values).all())
        sys.stderr.write(
            f'compare vs {os.path.basename(kag_matches[0])} | n={n} '
            f'smiles_aligned={same_smiles} pred_agreement={pred_agree:.4f} '
            f'prob_max_abs_diff={prob_mad:.6f} logit_max_abs_diff={logit_mad:.6f}\n'
        )
    else:
        sys.stderr.write('no kaggle test_predictions.csv found for comparison\n')


if __name__ == '__main__':
    main()
