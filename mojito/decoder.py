import torch
from torch import nn
import dgl

class Decoder(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        depth: int = 3,
        num_classes: int = 28,
        activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                torch.nn.Linear(
                    hidden_features if idx > 0 else in_features, 
                    hidden_features,
                )
                for idx in range(depth)
            ]
        )
        self.out = torch.nn.Linear(hidden_features, num_classes + hidden_features)
        self.depth = depth
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.activation = activation
        
    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)
        x = self.out(x)
        structure, embedding = x.split(
            [self.num_classes, self.hidden_features], dim=-1
        )
        
        return structure, embedding
    
    def loss(self, g, x):
        structure, embedding = self(x)
        loss_embedding = torch.distributions.Categorical(
            logits=embedding
        ).log_prob(g.ndata["type"].argmax(-1)).mean().mul(-1)
        accuracy_embedding = (embedding.argmax(-1) == g.ndata["type"].argmax(-1)).float().mean()
        
        num_atoms = g.batch_num_nodes()
        mask = torch.repeat_interleave(
            torch.arange(len(num_atoms), device=structure.device), num_atoms
        )
        mask = (mask.unsqueeze(0) == mask.unsqueeze(1))
        
        total_atoms = num_atoms.sum()
        structure = structure @ structure.t()
        structure = structure * (1 - torch.eye(total_atoms, device=structure.device))
        adj = torch.zeros(total_atoms, total_atoms, device=structure.device)
        adj[g.edges()[0], g.edges()[1]] = 1
        adj.fill_diagonal_(0)
        loss_structure = torch.distributions.Bernoulli(
            logits=structure,
        ).log_prob(adj).mul(-1)
        loss_structure = loss_structure[mask].mean()
        accuracy_structure = (structure.sigmoid().round() == adj)[mask].float().mean()
        return loss_embedding, loss_structure, accuracy_embedding, accuracy_structure

            
        
        