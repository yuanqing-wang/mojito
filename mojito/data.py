from typing import List
from collections import defaultdict
import random
from functools import lru_cache
from rdkit import Chem
import torch



def from_smiles(smiles: str):
    """Convert SMILES to adjacency matrix and node features."""
    mol = Chem.MolFromSmiles(smiles)
    a = torch.tensor(Chem.rdmolops.GetAdjacencyMatrix(mol), dtype=torch.float32)
    h = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.float32)
    h = torch.nn.functional.one_hot(h.long(), num_classes=119).float()  # 119 is the number of atomic numbers in RDKit
    return a, h

@lru_cache
def adj_power(a: torch.Tensor, power: int = 8):
    return torch.stack(
        [
            torch.matrix_power(a, i+1) for i in range(power)
        ],
        dim=-1,
    )
    
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, power: int = 8):
        self.data = graphs
        self.power = power

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        a, h = self.data[idx]
        return adj_power(a, power=self.power), h
    
    @classmethod
    def from_smiles(cls, smiles: List[str], power: int = 8):
        """Create a dataset from a SMILES string."""
        graphs = [from_smiles(s) for s in smiles]
        return cls(graphs, power=power)

class GraphSampler(torch.utils.data.Sampler):
    def __init__(
            self, 
            dataset: GraphDataset, 
            batch_size: int = 32,
            shuffle: bool = True,
        ):
        self.length_to_indices = defaultdict(list)
        for idx, (a, h) in enumerate(dataset):
            self.length_to_indices[len(a)].append(idx)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches = self._create_batches()

    def _create_batches(self):
        batches = []
        for indices in self.length_to_indices.values():
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        return batches

    def __iter__(self):
        if self.shuffle:
            self.batches = self._create_batches()
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)