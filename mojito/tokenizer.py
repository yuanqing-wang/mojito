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
        
    def encode(self, g, x):
        return self.encoder(g, x)
    
    forward = encode
    
    def decode(self, x):
        return self.decoder(x)
    
    def reconstruction_loss(self, g, x):
        return self.decoder.loss(g, self.encode(g, x))
        
        