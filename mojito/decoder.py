import torch
from torch import nn
import dgl

class Decoder(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        depth: int = 3,
        activation_fn: nn.Module = nn.Tanh(),
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
        
        self.depth = depth
        
        self.activation_fn = activation_fn
        
    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < self.depth - 1:
                x = self.activation_fn(x)
        return x
    
    def _loss_one_graph(self, g, x):
        n_nodes = g.number_of_nodes()
        adj_hat = x @ x.t()
        adj_hat = adj_hat.sigmoid()
        adj_hat = adj_hat * (1 - torch.eye(n_nodes))
        adj = torch.zeros(n_nodes, n_nodes)
        adj[g.edges()[0], g.edges()[1]] = 1
        adj.fill_diagonal_(0)
        loss = -torch.distributions.Bernoulli(adj_hat).log_prob(adj).mean()
        accuracy = (adj_hat.round() == adj).float().mean()
        print(accuracy)
        return loss
    
    def loss(self, g, x):
        x = self(x)
        if g.batch_size == 1:
            return self._loss_one_graph(g, x)
        else:
            g.ndata["x"] = x
            graphs = dgl.unbatch(g)
            losses = 0.0
            for graph in graphs:
                losses += self._loss_one_graph(graph, graph.ndata["x"])
            return losses
            
        
        