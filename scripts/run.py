import torch
import dgl

def one_hot(g):
    g = dgl.remove_self_loop(g)
    if g.ndata["feat"].dtype == torch.int64:
        g.ndata["feat"] = torch.nn.functional.one_hot(
            g.ndata["feat"], num_classes=28,
        ).float()
    return g

def run():
    from mojito import Encoder, Decoder, Tokenizer
    from dgl.data import ZINCDataset
    dataset = ZINCDataset(transform=one_hot)
    tokenizer = Tokenizer(
        encoder=Encoder(28, 32),
        decoder=Decoder(32, 32),
    )
    
    if torch.cuda.is_available():
        tokenizer = tokenizer.cuda()
    
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)
        
    for _ in range(1000):
        for g, _ in dataset:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                g = g.to("cuda")
            loss = tokenizer.reconstruction_loss(g, g.ndata["feat"])
            loss.backward()
            optimizer.step()
    
        
if __name__ == "__main__":
    run()