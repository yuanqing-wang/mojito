from dataclasses import dataclass
from collections import namedtuple
from functools import reduce  
from typing import Sequence, Optional
from rdkit import Chem
import networkx as nx

def smiles(fn):
    def wrapper(*args, **kwargs):
        if isinstance(args[0], str):
            args = (Chem.MolFromSmiles(args[0]),) + args[1:]
        return fn(*args, **kwargs)
    wrapper.__doc__ = fn.__doc__
    return wrapper

@dataclass
class Bijection:
    old: Optional[Sequence] = None
    new: Optional[Sequence] = None
    
    def __post_init__(self):
        if self.old is None:
            self.old = []
        if self.new is None:
            self.new = []
        assert len(self.old) == len(self.new)
        self.old_to_new = {o: n for o, n in zip(self.old, self.new)}
        self.new_to_old = {n: o for o, n in zip(self.old, self.new)}
        
    def add(self, old, new):
        self.old.append(old)
        self.new.append(new)
        self.old_to_new[old] = new
        self.new_to_old[new] = old
        
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
    
    molecule: rdkit.Chem.Mol
        The original molecule from which the fragment is generated.
        
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
    to_delete = [atom.GetIdx() for atom in fragment.GetAtoms() if atom.GetSymbol() == "*"]
    fragment = Chem.EditableMol(fragment)
    for idx in to_delete[::-1]:
        fragment.RemoveAtom(idx)
    fragment = fragment.GetMol()
    
    # replace halogen with hydrogen
    for atom in fragment.GetAtoms():
        if atom.GetSymbol() in ["F", "Cl", "Br", "I"]:
            modification[atom.GetIntProp("_idx")] = str(atom.GetSymbol())
            atom.SetAtomicNum(1)
            
    # replace S with O if S has fewer than 2 bonds
    for atom in fragment.GetAtoms():
        if atom.GetSymbol() == "S" and len(atom.GetNeighbors()) <= 2:
            modification[atom.GetIntProp("_idx")] = "S"
            atom.SetAtomicNum(8)
    
    # reconstruct and canonicalize the fragment
    canonicalize_fragment = Chem.MolFromSmiles(
        Chem.MolToSmiles(fragment, canonical=True)
    )
    
    # get the mapping from the original fragment to the canonicalized fragment
    renumbering = fragment.GetSubstructMatch(canonicalize_fragment)    
    renumbering = [fragment.GetAtomWithIdx(i).GetIntProp("_idx") for i in renumbering]
    renumbering = Bijection(old=renumbering, new=list(range(len(renumbering))))

    # update the modification dictionary so that it contains
    # the canonical index as key
    modification = {
        renumbering.forward(old_idx): modification[old_idx]
        for old_idx in modification.keys()
    }
    return canonicalize_fragment, renumbering, modification

_LEFT = Chem.MolFromSmarts('[*!D1]-&!@[!$(*#*)&!D1]')
_RIGHT = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[*D1]')
def get_rotatable_bonds(molecule):
    return molecule.GetSubstructMatches(_LEFT) + \
        molecule.GetSubstructMatches(_RIGHT)

@smiles
def generate_fragments(molecule):
    """ Generate fragments by cutting rotatable bonds.
    
    Parameters
    ----------
    molecule: rdkit.Chem.Mol
        The molecule to be fragmented.
        
    Returns
    -------
    fragments: list of rdkit.Chem.Mol
        The list of fragments.
        
    renumbering: list of Bijection
        The mapping from the original fragment to the canonicalized fragment.
        
    modifications: list of dict
        The mapping from the canonical index to the original atom number.
        
    rotatable_bond_idxs: list of rdkit.Chem.Bond
        The list of rotatable bonds that were cut to generate the fragments.
        
    """
    for atom in molecule.GetAtoms():
        atom.SetIntProp("_idx", atom.GetIdx())
    rotatable_bonds = get_rotatable_bonds(molecule)
    rotatable_bond_idxs = [molecule.GetBondBetweenAtoms(*bond) for bond in rotatable_bonds]
    fragments = Chem.FragmentOnBonds(molecule, [bond.GetIdx() for bond in rotatable_bond_idxs], addDummies=True)
    fragments = Chem.GetMolFrags(fragments, asMols=True, sanitizeFrags=False)            
    fragments, renumbering, modifications = zip(*[canonicalize_fragment(frag) for frag in fragments])
    fragments = [Chem.MolToSmiles(frag, canonical=True) for frag in fragments]
    return fragments, renumbering, modifications, rotatable_bonds
    
@smiles
def molecule_to_tree(molecule):
    """ Build a tree from the fragments of a molecule.
    
    Parameters
    ----------
    molecule: rdkit.Chem.Mol or str
        The molecule to be fragmented and converted to a tree.
        
    Returns
    -------
    tree: networkx.Graph
        The tree representation of the molecule.
        
    Examples
    --------
    
    """
    fragments, renumbering, modifications, rotatable_bonds = generate_fragments(molecule)
    tree = nx.Graph()
    for idx, fragment in enumerate(fragments):
        tree.add_node(
            idx,
            fragment=fragment,
            modifications=modifications[idx],
        )
        
    for old_src, old_dst in rotatable_bonds:
        for idx, bijection in enumerate(renumbering):
            if old_src in bijection.old:
                src_global = idx
                src_local = bijection.forward(old_src)
            if old_dst in bijection.old:
                dst_global = idx
                dst_local = bijection.forward(old_dst)
        tree.add_edge(
            src_global,
            dst_global,
            src_idx=src_local,
            dst_idx=dst_local,
        )
        
    return tree

def tree_to_molecule(tree):
    """ Reconstruct a molecule from its tree representation.
    
    Parameters
    ----------
    tree: networkx.Graph
        The tree representation of the molecule.
        
    Returns
    -------
    molecule: rdkit.Chem.Mol
        The reconstructed molecule.
        
    Examples
    --------
    >>> tree = molecule_to_tree("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
    >>> molecule = tree_to_molecule(tree)
    
    
    """
    # record the origin of each atom in the final molecule
    fragments = []
    for idx, data in tree.nodes(data=True):
        fragment = Chem.MolFromSmiles(data["fragment"])
        # modification = data["modification"]
        for atom in fragment.GetAtoms():
            atom.SetIntProp("_global_idx", idx)
            atom.SetIntProp("_local_idx", atom.GetIdx())
        fragments.append(fragment)
    
    # combine the fragment into one molecule
    molecule = reduce(Chem.CombineMols, fragments)
    molecule = Chem.EditableMol(molecule)
    
    # loop through edges to add bonds
    for src_global, dst_global, data in tree.edges(data=True):
        src_local = data["src_idx"]
        dst_local = data["dst_idx"]
        src_atom = None
        dst_atom = None
        for atom in molecule.GetMol().GetAtoms():
            if atom.GetIntProp("_global_idx") == src_global and atom.GetIntProp("_local_idx") == src_local:
                src_atom = atom
            if atom.GetIntProp("_global_idx") == dst_global and atom.GetIntProp("_local_idx") == dst_local:
                dst_atom = atom
        assert src_atom is not None and dst_atom is not None
        molecule.AddBond(src_atom.GetIdx(), dst_atom.GetIdx(), order=Chem.rdchem.BondType.SINGLE)
        
    molecule = molecule.GetMol()
    return molecule
    
class Tokenizer:
    dictionary = {}
    
    def process(self, molecule):
        if isinstance(molecule, str):
            molecule = Chem.MolFromSmiles(molecule)
        
        