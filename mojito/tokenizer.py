import torch

class Tokenizer(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        quantizer: torch.nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer

    def encode(self, a, x):
        return self.encoder(a, x)
    
    forward = encode
    
    def decode(self, x):
        return self.decoder(x)

    def quantize(self, x):
        return self.quantizer(x)

    def loss(self, a, x):
        x0 = x
        x = self.encode(a, x)

        xq, idxs, loss_quantization = self.quantize(x)
        structure, embedding = self.decode(xq)
        loss_embedding = torch.distributions.Categorical(
            logits=embedding
        ).log_prob(x0.argmax(-1)).mean().mul(-1)
        accuracy_embedding = (embedding.argmax(-1) == x0.argmax(-1)).float().mean()
        
        adj = a[..., 0] > 0 

        structure = structure @ structure.swapaxes(-1, -2)
        structure = structure * (1 - torch.eye(x.shape[-2], device=structure.device))
        # print(structure)
        
        
        loss_structure = torch.distributions.Bernoulli(
            logits=structure,
        ).log_prob(adj.float()).mul(-1)# .mean()
        loss_structure = loss_structure.mean()
        
        
        accuracy_structure = (structure.gt(0) == adj).float().mean()

        return loss_quantization, loss_embedding, loss_structure, accuracy_embedding, accuracy_structure
