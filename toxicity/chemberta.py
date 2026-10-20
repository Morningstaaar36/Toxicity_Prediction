import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ChemBERTaEmbedder(nn.Module):
    def __init__(self, model_name='seyonec/ChemBERTa-zinc-base-v1', freeze=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.embedding_dim = self.model.config.hidden_size

    def forward(self, smiles_list):
        inputs = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.set_grad_enabled(self.training):
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings
