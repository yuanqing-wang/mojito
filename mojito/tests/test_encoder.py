import torch
import dgl

def test_forward():
    from mojito.encoder import Encoder
    g = dgl.rand_graph(10, 20)
    x = torch.randn(10, 32)
    encoder = Encoder(32, 64)
    h = encoder(g, x)
    assert h.shape == (10, 64)
    
    