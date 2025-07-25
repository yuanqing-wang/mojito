import random
import torch

def test_data():
    from mojito.data import GraphDataset, GraphSampler
    graphs = []
    for idx in range(5, 10):
        for _ in range(10):
            a = torch.randn(idx, idx)
            h = torch.randn(idx, 32)
            graphs.append((a, h))
    random.shuffle(graphs)
    dataset = GraphDataset(graphs)
    sampler = GraphSampler(dataset, batch_size=2, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
    )
    
    for batch in dataloader:
        a, h = batch
        print(f"Batch size: {len(a)}")
        print(f"Adjacency matrix shape: {a.shape}")
        print(f"Node features shape: {h.shape}")
        assert a.shape[0] == h.shape[0], "Batch size mismatch between adjacency and node features."
        
        
def test_smiles():
    from mojito.data import from_smiles
    smiles = "CCO"
    a, h = from_smiles(smiles)
    print(f"Adjacency matrix shape: {a.shape}")
    print(f"Node features shape: {h.shape}")
    
        
if __name__ == "__main__":
    test_smiles()
    