from turtle import pd
import torch
from torch import nn

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
    

            
        
        
