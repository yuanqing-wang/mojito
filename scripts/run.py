import torch
import dgl

import wandb
wandb.login(
    key="58466296c2de2fdd61d262115503afdf302441b7",
)
from datetime import datetime
name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
wandb.init(
    project="mojito",
    name=name,
)

def one_hot(g):
    g = dgl.remove_self_loop(g)
    if g.ndata["feat"].dtype == torch.int64:
        g.ndata["type"] = torch.nn.functional.one_hot(
            g.ndata["feat"], num_classes=28,
        ).float()
        
        g.ndata["feat"] = torch.cat(
            [
                g.ndata["type"],
                dgl.laplacian_pe(g, 8),
            ],
            dim=-1,
        )
    return g

def run():
    from mojito import Encoder, Decoder, Tokenizer
    from dgl.data import ZINCDataset
    dataset = ZINCDataset(transform=one_hot)
    dataloader = dgl.dataloading.GraphDataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
    )
    tokenizer = Tokenizer(
        encoder=Encoder(28+8, 1024),
        decoder=Decoder(256, 1024),
    )
    
    if torch.cuda.is_available():
        tokenizer = tokenizer.cuda()
    
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)
        
    for _ in range(1000):
        for g, _ in dataloader:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                g = g.to("cuda")
            (
                loss_embedding,
                loss_structure,
                accuracy_embedding,
                accuracy_structure,    
            ) = tokenizer.reconstruction_loss(g, g.ndata["feat"])
            loss = loss_embedding + loss_structure
            loss.backward()
            optimizer.step()
    
            wandb.log({
                "loss_embedding": loss_embedding.item(),
                "loss_structure": loss_structure.item(),
                "accuracy_embedding": accuracy_embedding.item(),
                "accuracy_structure": accuracy_structure.item(),
            })
        
if __name__ == "__main__":
    run()