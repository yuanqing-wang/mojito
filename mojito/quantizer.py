import torch

class Quantizer(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_features: int,
        beta: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.beta = beta
        self.weight = torch.nn.Parameter(torch.empty(num_classes, hidden_features))        
        torch.nn.init.uniform_(self.weight, -1/num_classes, 1/num_classes)
        
    def gradually_detach(self, x, t):
        if t >= 1.0:
            return x.detach()
        return (1 - t) * x + t * x.detach()

    def forward(self, x, t):
        weight = self.weight  # (num_classes, hidden_features)
        distance = x.unsqueeze(-2) - weight  # (batch, num_nodes, num_classes, hidden_features)
        distance = (distance ** 2).sum(-1)  # (batch, num_nodes, num_classes)
        idxs = distance.argmin(-1)  # (batch, num_nodes)
        xq = torch.nn.functional.one_hot(idxs, self.num_classes).float()
        xq = xq @ weight  # (batch, num_nodes, hidden_features)
        loss = (xq.detach() - x).pow(2).mean() + self.beta * (xq - x.detach()).pow(2).mean()
        xq = x + self.gradually_detach(xq - x, t)  # straight-through estimator
        return xq, idxs, loss
        
