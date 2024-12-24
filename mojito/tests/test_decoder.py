import torch
import dgl

def test_forward():
    from mojito.decoder import Decoder
    g = dgl.rand_graph(10, 20)
    x = torch.randn(10, 32)
    decoder = Decoder(32, 64)
    loss = decoder.loss(g, x)
    print(loss)