import torch
import torch.nn as nn
from functools import partial

class Layer(nn.Module):
    def __init__(
        self,
        hidden_features: int,
        num_heads: int = 8,
        power: int = 8,
        activation: torch.nn.Module = torch.nn.SiLU(),
    ):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=hidden_features,
            num_heads=num_heads,
            batch_first=True,
        )
        self.fc_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, hidden_features),
        )
        self.power_to_head = torch.nn.Sequential(
            torch.nn.Linear(power, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, num_heads),
        )
        
    def forward(
        self,
        a: torch.Tensor,
        h: torch.Tensor,
    ):
        a = self.power_to_head(a)
        h, _ = self.mha(h, h, h, attn_mask=a.moveaxis(-1, -3))
        h = self.fc_out(h)
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
        num_heads: int = 8,
        power: int = 8,
        activation: nn.Module = nn.SiLU(),
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
                Layer(
                    hidden_features,
                    num_heads=num_heads,
                    power=power,
                    activation=activation
                )
                for _ in range(depth)
            ]
        )
                
    def forward(self, a, h):
        h = self.fc_in(h)
        for layer in self.layers:
            h = layer(a, h)
        h = h.tanh()
        return h
