from calendar import c
import pandas as pd
from math import ceil
from mojito.tokenizer import build_library, molecule_to_tree, tree_to_molecule, tree_to_string
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import tqdm
from rdkit.Chem import Draw

def smiles_to_pdf(
    smiles_list,
    out_pdf="molecules.pdf",
    legends=None,
    mols_per_row=5,
    subimg_size=(250, 250),
    max_rows_per_page=7,  # adjust to control page height
    kekulize=True
):
    """
    Render a list of SMILES to a (possibly multi-page) PDF.

    Args:
        smiles_list (list[str]): SMILES strings.
        out_pdf (str): Output PDF path.
        legends (list[str] | None): Captions under each molecule (same length as smiles_list).
        mols_per_row (int): Number of molecules per row.
        subimg_size (tuple[int, int]): Size (w,h) of each cell in pixels.
        max_rows_per_page (int): Maximum grid rows per PDF page before starting a new page.
        kekulize (bool): Kekulize molecules (often improves look).
    """
    legends = legends or ["" for _ in smiles_list]

    # Build and prep molecules
    mols = []
    kept_legends = []
    for smi, legend in zip(smiles_list, legends):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            # skip invalid SMILES but continue
            continue
        # m = Chem.AddHs(m, addCoords=True)  # helps 2D layout in some cases
        AllChem.Compute2DCoords(m)
        if kekulize:
            try:
                Chem.Kekulize(m, clearAromaticFlags=True)
            except Exception:
                pass  # fall back if kekulization fails
        mols.append(m)
        kept_legends.append(legend)

    if not mols:
        raise ValueError("No valid molecules were parsed from the provided SMILES.")

    # Page/chunking
    cells_per_page = mols_per_row * max_rows_per_page
    num_pages = ceil(len(mols) / cells_per_page)

    pil_pages = []
    for i in range(num_pages):
        start = i * cells_per_page
        end = min((i + 1) * cells_per_page, len(mols))
        mol_chunk = mols[start:end]
        legend_chunk = kept_legends[start:end]

        img = Draw.MolsToGridImage(
            mol_chunk,
            molsPerRow=mols_per_row,
            subImgSize=subimg_size,
            legends=legend_chunk,
            useSVG=False,  # returns a PIL Image
        )
        pil_pages.append(img.convert("RGB"))

    # Save single or multi-page PDF
    if len(pil_pages) == 1:
        pil_pages[0].save(out_pdf, "PDF")
    else:
        pil_pages[0].save(out_pdf, "PDF", save_all=True, append_images=pil_pages[1:])

    return out_pdf

def run():
    print(len("CN1C=NC2=C1C(=O)N(C(=O)N2C)C"))
    print(tree_to_string(molecule_to_tree("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")))
    
    URL = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
    df = pd.read_csv(URL)["smiles"].tolist()
    errors = []
    
    
    for smiles in tqdm.tqdm(df):
            smiles = smiles.strip()
            molecule = Chem.MolFromSmiles(smiles)
            Chem.RemoveStereochemistry(molecule)
            smiles = Chem.MolToSmiles(molecule, canonical=True, kekuleSmiles=False)
            
            tree = molecule_to_tree(smiles)
            
            new_molecule = tree_to_molecule(tree)
            new_smiles = Chem.MolToSmiles(new_molecule, canonical=True, kekuleSmiles=False)

            if smiles != new_smiles:
                errors.append((smiles, new_smiles))

    for i, (s1, s2) in enumerate(errors):
        print(f"{i}: {s1} -> {s2}")


    
    
    
    
if __name__ == "__main__":
    run()