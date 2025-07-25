import dgl
import torch
import torch.nn as nn
from functools import partial

GCN = partial(dgl.nn.GraphConv, allow_zero_in_degree=True)
GAT = partial(dgl.nn.GATConv, num_heads=1)
GraphSAGE = partial(dgl.nn.SAGEConv, aggregator_type='mean')

def get_attn_mask(graph: dgl.DGLGraph):
    number_of_nodes = graph.batch_num_nodes()
    graph_idx = torch.repeat_interleave(torch.arange(len(number_of_nodes), device=graph.device), number_of_nodes)
    binary_mask = graph_idx.unsqueeze(0) != graph_idx.unsqueeze(1)
    adj = graph.adj().to_dense()
    return binary_mask, adj

class GIN(dgl.nn.GINConv):
    def __init__(self, in_feats, out_feats):
        super().__init__(
            apply_func=torch.nn.Linear(in_feats, out_feats),
            aggregator_type='sum',
            learn_eps=True,
        )
        

class Attention(torch.nn.Module):
    def __init__(
        self,
        hidden_features: int,
        num_heads: int,
        activation: torch.nn.Module = torch.nn.SiLU(),
        dropout: float = 0.1,
        power: int = 10,
    ):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=hidden_features,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_features, hidden_features),
        )
        
        self.activation = activation
        self.norm0 = torch.nn.LayerNorm(hidden_features)
        self.norm1 = torch.nn.LayerNorm(hidden_features)
        self.power_to_head = torch.nn.Sequential(
            torch.nn.Linear(power, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, num_heads),
        )
        self.power = power
        self.INF = 1e9
        
    def forward(
        self,
        g: dgl.DGLGraph,
        h: torch.Tensor,
    ):
        g = g.local_var()
        mask, adj = get_attn_mask(g)
        adj = torch.stack([torch.matrix_power(adj, i+1) for i in range(self.power)], dim=-1)
        adj = self.power_to_head(adj)
        adj = adj - mask.unsqueeze(-1) * self.INF
        
        h0 = h
        h = self.norm0(h)
        h = self.mha(h, h, h, attn_mask=mask)[0] + h0
        h0 = h
        h = self.norm1(h)
        h = self.ffn(h) + h0
        return h

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
        num_heads: int = 4,
        layer: torch.nn.Module = GIN,
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
                
        self.attentions = nn.ModuleList(
            [
                Attention(hidden_features, num_heads=num_heads)
                for _ in range(depth)
            ]
        )
                       
        self.activation_fn = activation_fn
        self.hidden_features = hidden_features
        self.depth = depth

    def forward(self, g, h):
        h = self.fc_in(h)
        for idx, layer in enumerate(self.layers):
            h0 = h
            h = layer(g, h)
            h = self.attentions[idx](g, h)
            if idx < self.depth - 1:
                h = self.activation_fn(h)
            h = h + h0
        h = h.tanh()
        return h
