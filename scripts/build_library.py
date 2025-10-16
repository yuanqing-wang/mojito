import pandas as pd
from mojito.utils import build_library

def run():
    URL = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
    df = pd.read_csv(URL)["smiles"].tolist()# [:1000]
    library = build_library(df)
    
    # save the dictionary to json
    import json
    with open("library.json", "w") as f:
        json.dump(library, f)
    
if __name__ == "__main__":
    run()