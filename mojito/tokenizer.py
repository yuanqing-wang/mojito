import torch

class Tokenizer(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def encode(self, a, x):
        return self.encoder(a, x)
    
    forward = encode
    
    def decode(self, x):
        return self.decoder(x)
    
    def loss(self, a, x):
        x0 = x
        x = self.encode(a, x)
        structure, embedding = self.decode(x)
        loss_embedding = torch.distributions.Categorical(
            logits=embedding
        ).log_prob(x0.argmax(-1)).mean().mul(-1)
        accuracy_embedding = (embedding.argmax(-1) == x0.argmax(-1)).float().mean()
        
        adj = a[..., 0]
        structure = structure @ structure.swapaxes(-1, -2)
        structure = structure * (1 - torch.eye(x.shape[-2], device=structure.device))
        
        # print(structure.shape, adj.shape)
        loss_structure = torch.distributions.Bernoulli(
            logits=structure,
        ).log_prob(adj).mul(-1).mean()
        accuracy_structure = (structure.sigmoid().round() == adj).float().mean()
        return loss_embedding, loss_structure, accuracy_embedding, accuracy_structure
        
        