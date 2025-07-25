import torch

def test_forward():
    from mojito.decoder import Decoder
    a = torch.randn(10, 10, 8).sigmoid().round()
    x = torch.randn(10, 32)
    decoder = Decoder(32, 64)
    loss = decoder.loss(a, x)