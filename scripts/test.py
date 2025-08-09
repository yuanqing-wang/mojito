import struct
from mojito import Encoder, Decoder, Tokenizer
from mojito.data import from_smiles, adj_power
import torch

def run():
    a, h = from_smiles("CCO")
    a = adj_power(a)
    tokenizer = Tokenizer(
        encoder=Encoder(119, 256),
        decoder=Decoder(256, 256, num_classes=119),
    )
    
    # x = tokenizer.encode(a, h)
    # structure, embedding = tokenizer.decode(x)
    
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)
    for _ in range(1000):
        optimizer.zero_grad()
        loss = tokenizer.loss(a, h)
        loss_embedding, loss_structure, accuracy_embedding, accuracy_structure = loss
        loss = loss_embedding + loss_structure
        loss.backward()
        optimizer.step()
        # print(
        #     f"Loss: {loss.item()}, "
        #     f"Loss Embedding: {loss_embedding.item()}, "
        #     f"Loss Structure: {loss_structure.item()}, "
        #     f"Accuracy Embedding: {accuracy_embedding.item()}, "
        #     f"Accuracy Structure: {accuracy_structure.item()}"
        # )
    
    
    
    
if __name__ == "__main__":
    run()