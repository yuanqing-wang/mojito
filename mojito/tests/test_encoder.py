import torch

def test_forward():
    from mojito.encoder import Encoder
    a = torch.randn(10, 10, 8)
    x = torch.randn(10, 32)
    encoder = Encoder(32, 64)
    h = encoder(a, x)
    assert h.shape == (10, 64)
    
    