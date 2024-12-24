import dgl
import torch
import torch.nn as nn
from functools import partial

GCN = partial(dgl.nn.GraphConv, allow_zero_in_degree=True)
GAT = partial(dgl.nn.GATConv, num_heads=1)
GraphSAGE = partial(dgl.nn.SAGEConv, aggregator_type='mean')

class Encoder(nn.Module):
    """Stacks of DGL layers.

    Parameters
    ----------
    layer : str
        The name of the DGL layer to use.

    prediction_head_params : dict
        Parameters for the prediction head.

    in_features : int
        The input dimension.

    hidden_features : int
        The model dimension.

    depth : int
        The number of layers.

    activation_fn : nn.Module
        The activation function to use.

    name : str
        The name of the model.

    Returns
    -------
    predictions : torch.Tensor
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        depth: int = 3,
        layer: torch.nn.Module = GraphSAGE,
        activation_fn: nn.Module = nn.SiLU(),
):
        super().__init__()

        # projection in
        self.fc_in = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_features, hidden_features),
        )

        # list of DGL layers
        self.layers = nn.ModuleList(
            [
                layer(
                    hidden_features, hidden_features,
                )
                for _ in range(depth)
            ]
        )
                
        self.activation_fn = activation_fn
        self.hidden_features = hidden_features
        self.depth = depth

    def forward(self, g, h):
        h = self.fc_in(h)
        for idx, layer in enumerate(self.layers):
            h = layer(g, h)
            if idx < self.depth - 1:
                h = self.activation_fn(h)
        h = h.tanh()
        return h
