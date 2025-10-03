from dataclasses import dataclass
from turtle import forward
from typing import Sequence
from rdkit import Chem

def smiles(fn):
    def wrapper(*args, **kwargs):
        if isinstance(args[0], str):
            args = (Chem.MolFromSmiles(args[0]),) + args[1:]
        return fn(*args, **kwargs)
    return wrapper

@dataclass
class Bijection:
    old: Sequence
    new: Sequence
    
    def __post_init__(self):
        assert len(self.old) == len(self.new)
        self.old_to_new = {o: n for o, n in zip(self.old, self.new)}
        self.new_to_old = {n: o for o, n in zip(self.old, self.new)}
        
    def forward(self, x):
        if isinstance(x, Sequence):
            return x.__class__(self.old_to_new[i] for i in x)
        return self.old_to_new[x]
    
    def backward(self, y):
        if isinstance(y, Sequence):
            return y.__class__(self.new_to_old[i] for i in y)
        return self.new_to_old[y]
    
@smiles
def canonicalize_fragment(fragment):    
    """ Map a fragment to its canonical form.
    
    Parameters
    ----------
    fragment: rdkit.Chem.Mol
        The fragment to be canonicalized.
        
    Returns
    -------
    canonical: rdkit.Chem.Mol
        The canonicalized fragment.
        
    mapping: tuple
        The mapping from the original fragment to the canonicalized fragment.
        
    modification: dict
        The mapping from the canonical index to the original atom number.
    """
    modification = {}
    
    # remove dummy atoms
    for atom in fragment.GetAtoms():
        if atom.GetSymbol() == "*":
            atom.SetAtomicNum(1)
    fragment = Chem.RemoveHs(fragment)
    
    # replace halogen with hydrogen
    for atom in fragment.GetAtoms():
        if atom.GetSymbol() in ["F", "Cl", "Br", "I"]:
            modification[atom.GetIdx()] = str(atom.GetSymbol())
            atom.SetAtomicNum(1)
            
    # replace S with O if S has fewer than 2 bonds
    for atom in fragment.GetAtoms():
        if atom.GetSymbol() == "S" and len(atom.GetNeighbors()) <= 2:
            modification[atom.GetIdx()] = "S"
            atom.SetAtomicNum(8)
    
    # reconstruct and canonicalize the fragment
    canonicalize_fragment = Chem.MolFromSmiles(
        Chem.MolToSmiles(fragment, canonical=True)
    )
    
    # get the mapping from the original fragment to the canonicalized fragment
    original_idxs = fragment.GetSubstructMatch(canonicalize_fragment)
    original_idxs = Bijection(old=original_idxs, new=range(len(original_idxs)))

    # update the modification dictionary so that it contains
    # the canonical index as key
    modification = {
        original_idxs.forward(old_idx): modification[old_idx]
        for old_idx in modification.keys()
    }
    return canonicalize_fragment, original_idxs, modification

_LEFT = Chem.MolFromSmarts('[*!D1]-&!@[!$(*#*)&!D1]')
_RIGHT = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[*D1]')
def get_rotatable_bonds(molecule):
    return molecule.GetSubstructMatches(_LEFT) + \
        molecule.GetSubstructMatches(_RIGHT)

@smiles
def generate_fragments(molecule):
    rotatable_bonds = get_rotatable_bonds(molecule)
    rotatable_bond_idxs = [molecule.GetBondBetweenAtoms(*bond) for bond in rotatable_bonds]
    fragments = Chem.FragmentOnBonds(molecule, [bond.GetIdx() for bond in rotatable_bonds], addDummies=True)
    fragments = Chem.GetMolFrags(fragments, asMols=True, sanitizeFrags=False)
    fragments, original_idxs, modifications = zip(*[canonicalize_fragment(frag) for frag in fragments])
    return fragments, original_idxs, modifications, rotatable_bond_idxs
    

    
class Tokenizer:
    dictionary = {}
    
    def process(self, molecule):
        if isinstance(molecule, str):
            molecule = Chem.MolFromSmiles(molecule)
        
        