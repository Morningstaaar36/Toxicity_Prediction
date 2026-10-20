from collections import Counter
from typing import List, Optional

import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


class MoleculeDataset(Dataset):

    def __init__(
        self,
        smiles_list: List[str],
        labels: Optional[List] = None,
        converter=None,
        pre_transform=None,
        transform=None,
        cache_graphs: bool = False
    ):
        self.smiles_list = smiles_list
        self.labels = labels
        self.converter = converter
        self.cache_graphs = cache_graphs
        self._cache = {}

        if labels is not None:
            assert len(smiles_list) == len(labels), \
                f"SMILES list ({len(smiles_list)}) and labels ({len(labels)}) must have same length"

        if converter is None:
            raise ValueError("Must provide a SMILESToPyG converter instance")

        super().__init__(root=None, transform=transform, pre_transform=pre_transform)

        if cache_graphs:
            for idx in tqdm(range(len(smiles_list)), desc="Caching graphs"):
                _ = self.get(idx)

    def len(self) -> int:
        return len(self.smiles_list)

    def get(self, idx: int) -> Data:
        if self.cache_graphs and idx in self._cache:
            return self._cache[idx]

        smiles = self.smiles_list[idx]

        try:
            data = self.converter.convert(smiles)

            if self.labels is not None:
                data.y = torch.tensor([self.labels[idx]], dtype=torch.float32)

            data.idx = idx

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if self.cache_graphs:
                self._cache[idx] = data

            return data

        except Exception as e:
            print(f"Error converting SMILES at index {idx} ('{smiles}'): {e}")
            raise

    def get_class_distribution(self) -> dict:
        if self.labels is None:
            return {}

        counter = Counter(self.labels)
        total = len(self.labels)

        distribution = {
            'counts': dict(counter),
            'percentages': {k: (v/total)*100 for k, v in counter.items()},
            'imbalance_ratio': max(counter.values()) / min(counter.values()) if len(counter) > 1 else 1.0
        }

        return distribution


def calculate_class_weights(labels: List) -> torch.Tensor:
    counter = Counter(labels)
    n_classes = len(counter)
    total_samples = len(labels)

    weights = {}
    for class_label, count in counter.items():
        weights[class_label] = total_samples / (n_classes * count)

    class_weights = torch.tensor([weights[i] for i in sorted(weights.keys())], dtype=torch.float32)

    return class_weights
