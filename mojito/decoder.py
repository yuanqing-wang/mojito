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
        adj_hat = x @ x.t()
        print(adj_hat.shape)
        n_nodes = g.number_of_nodes()
        adj = torch.zeros(n_nodes, n_nodes)
        adj[g.edges()[0], g.edges()[1]] = 1
        # nll_pos = -torch.distributions.Bernoulli(logits=adj_hat).log_prob(adj).mean()
        # nll_neg= -torch.distributions.Bernoulli(logits=adj_hat).log_prob(1 - adj).mean()
        loss = (adj- adj_hat.sigmoid()).pow(2).mean()
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
            
        
        