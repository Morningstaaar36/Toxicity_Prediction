import torch
import torch.nn as nn

from .chemberta import ChemBERTaEmbedder
from .fusion import BCMGFusion
from .gnn import GATEncoder


class ImprovedClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3, num_classes=1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hid_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hid_dim))
            layers.append(nn.LayerNorm(hid_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hid_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


class ToxicityPredictor(nn.Module):

    def __init__(
        self,
        node_feat_dim=44,
        edge_feat_dim=10,
        lap_pe_dim=8,
        hidden_dim=256,
        num_gnn_layers=3,
        num_heads=4,
        dropout=0.1,
        classifier_dropout=0.3,
        fusion_rank=128,
        fusion_out_dim=256,
        classifier_hidden_dims=[256, 128, 64],
        num_classes=1,
        freeze_chemberta=True,
        use_gatv2=True,
        chemberta_model='seyonec/ChemBERTa-zinc-base-v1'
    ):
        super().__init__()

        self.gat_encoder = GATEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            lap_pe_dim=lap_pe_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_gatv2=use_gatv2,
        )

        self.chemberta = ChemBERTaEmbedder(
            model_name=chemberta_model,
            freeze=freeze_chemberta
        )
        chemberta_dim = self.chemberta.embedding_dim
        graph_emb_dim = 2 * hidden_dim


        self.fusion = BCMGFusion(
            d_g=graph_emb_dim, d_l=chemberta_dim,
            d_out=fusion_out_dim, rank=fusion_rank,
        )

        self.proj_g = nn.Linear(graph_emb_dim, fusion_out_dim)
        self.proj_l = nn.Linear(chemberta_dim,  fusion_out_dim)

        self.classifier = ImprovedClassifier(
            input_dim=2 * fusion_out_dim,
            hidden_dims=classifier_hidden_dims,
            dropout=classifier_dropout,
            num_classes=num_classes
        )

        self.hidden_dim     = hidden_dim
        self.fusion_out_dim = fusion_out_dim
        self._init_weights()


    def forward(self, batch, return_embeddings_only: bool = False):

        d_g_raw, intermediate_pools = self.gat_encoder(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            lap_pe=batch.laplacian_pe,
            batch=batch.batch
        )
        d_l_raw = self.chemberta(batch.smiles)

        d_g_proj = self.proj_g(d_g_raw)
        d_l_proj = self.proj_l(d_l_raw)

        z, gate = self.fusion(d_g_raw, d_l_raw)

        if return_embeddings_only:
            return {
                'z'   : z,
                'd_g' : d_g_proj,
                'd_l' : d_l_proj,
                'gate': gate,
            }


        cls_input = torch.cat([d_g_proj, d_l_proj], dim=-1)
        logits    = self.classifier(cls_input)

        return {
            'logits'            : logits,
            'z'                 : z,
            'd_g'               : d_g_proj,
            'd_l'               : d_l_proj,
            'gate'              : gate,
            'intermediate_pools': intermediate_pools,
        }

    def predict(self, batch, threshold=0.5):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch)
            logits  = outputs['logits'].squeeze()
            probs   = torch.sigmoid(logits)
            preds   = (probs >= threshold).long()
        return preds, probs

    def _init_weights(self):
        for name, module in self.named_modules():
            if 'chemberta' in name:
                continue
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if module.weight is not None:
                    torch.nn.init.ones_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
