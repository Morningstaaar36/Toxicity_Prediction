import torch

from .losses import ContrastiveLossZG, ContrastiveLossZL
from .metrics import compute_metrics


def train_stage1_contrastive(model, train_loader, optimizer, device, num_epochs=25, temperature=0.5):
    criterion_zg = ContrastiveLossZG(temperature=temperature)
    criterion_zl = ContrastiveLossZL(temperature=temperature)

    model.train()
    print("\n" + "="*80)
    print("STAGE 1: CONTRASTIVE PRETRAINING  [L(z,d_g)+L(z,d_l)]")
    print("="*80)

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for batch in train_loader:
            batch    = batch.to(device)
            outputs  = model(batch, return_embeddings_only=True)
            z        = outputs['z']
            d_g_proj = outputs['d_g']
            d_l_proj = outputs['d_l']

            loss = criterion_zg(z, d_g_proj) + criterion_zl(z, d_l_proj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:>3}/{num_epochs} | Contrastive Loss: "
                  f"{total_loss/len(train_loader):.4f}")


def train_stage2_classification_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    all_preds = []
    all_scores = []
    all_labels = []

    for batch in train_loader:
        batch = batch.to(device)

        outputs = model(batch, return_embeddings_only=False)
        logits = outputs['logits']
        targets = batch.y

        loss = criterion(logits.view(-1), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        labels = targets.detach().cpu().numpy()

        all_preds.extend(preds.flatten())
        all_scores.extend(probs.flatten())
        all_labels.extend(labels.flatten())

    metrics = compute_metrics(all_labels, all_preds, all_scores)
    metrics['loss'] = total_loss / len(train_loader)

    return metrics


@torch.no_grad()
def evaluate_stage2(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    all_preds = []
    all_scores = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)

        outputs = model(batch, return_embeddings_only=False)
        logits = outputs['logits']
        targets = batch.y

        loss = criterion(logits.view(-1), targets.view(-1))
        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        labels = targets.cpu().numpy()

        all_preds.extend(preds.flatten())
        all_scores.extend(probs.flatten())
        all_labels.extend(labels.flatten())

    metrics = compute_metrics(all_labels, all_preds, all_scores)
    metrics['loss'] = total_loss / len(loader)

    return metrics
