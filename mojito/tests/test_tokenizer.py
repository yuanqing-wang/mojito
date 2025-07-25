import torch

def test_construct():
    from mojito.encoder import Encoder
    from mojito.decoder import Decoder
    from mojito.tokenizer import Tokenizer
    
    encoder = Encoder(32, 64)
    decoder = Decoder(64, 64)
    tokenizer = Tokenizer(encoder, decoder)
    
    a = torch.randn(10, 10, 8).sigmoid().round()
    h = torch.randn(10, 32)
    loss = tokenizer.reconstruction_loss(a, h)
    print(loss)