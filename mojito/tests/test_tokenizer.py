import torch
import dgl

def test_construct():
    from mojito.encoder import Encoder
    from mojito.decoder import Decoder
    from mojito.tokenizer import Tokenizer
    
    encoder = Encoder(32, 64)
    decoder = Decoder(64, 64)
    tokenizer = Tokenizer(encoder, decoder)
    
    g = dgl.rand_graph(10, 20)
    g.ndata["h"] = torch.randn(10, 32)
    loss = tokenizer.reconstruction_loss(g, g.ndata["h"])
    print(loss)