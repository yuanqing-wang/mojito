import torch
import pandas as pd

import wandb
wandb.login(
    key="58466296c2de2fdd61d262115503afdf302441b7",
)
from datetime import datetime
name = "control" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
wandb.init(
    project="mojito",
    name=name,
)


def run():
    from mojito import Encoder, Decoder, Tokenizer, Quantizer
    from mojito.data import GraphDataset, GraphSampler
    
    URL = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
    # URL = "250k_rndm_zinc_drugs_clean_3.csv"
    df = pd.read_csv(URL)
    
    # df = pd.read_csv("250k_rndm_zinc_drugs_clean_3.csv", nrows=100)
    smiles = df["smiles"].tolist()
    dataset = GraphDataset.from_smiles(smiles, power=8)
    sampler = GraphSampler(dataset, batch_size=8, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
    )

    tokenizer = Tokenizer(
        encoder=Encoder(119, 128),
        decoder=Decoder(128, 128, num_classes=119),
        quantizer=Quantizer(num_classes=10240, hidden_features=128)
    )
    
    if torch.cuda.is_available():
        tokenizer = tokenizer.cuda()
    
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3, weight_decay=1e-5)
        
    for idx in range(1000000):
        t = min(1.0, float(idx) / 10000)
        for a, h in dataloader:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                a, h = a.to("cuda"), h.to("cuda")
            (
                loss_quantization,
                loss_embedding,
                loss_structure,
                accuracy_embedding,
                accuracy_structure,    
            ) = tokenizer.loss(a, h, t)
            loss = loss_structure + loss_embedding + loss_quantization
            loss.backward()
            optimizer.step()
            
            wandb.log({
                # "loss_quantization": loss_quantization.item(),
                "loss_embedding": loss_embedding.item(),
                "loss_structure": loss_structure.item(),
                "accuracy_embedding": accuracy_embedding.item(),
                "accuracy_structure": accuracy_structure.item(),
            })
        
if __name__ == "__main__":
    run()
