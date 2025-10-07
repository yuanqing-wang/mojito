from dataclasses import dataclass
from collections import defaultdict, namedtuple
from functools import reduce  
from typing import Sequence, Optional
from rdkit import Chem
import networkx as nx
import tqdm

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
    modification = defaultdict(list)
    
    # remove dummy atoms    
    for atom in fragment.GetAtoms():
        if atom.GetSymbol() == "*":
            atom.SetAtomicNum(1)
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)
    
    # replace halogen with hydrogen
    fragment = Chem.RWMol(fragment)
    Chem.Kekulize(fragment)
    to_delete = []
    for atom in fragment.GetAtoms():
        if atom.GetSymbol() in ["F", "Cl", "Br", "I"]:
            # get the _idx of the neighbor atom
            neighbor = atom.GetNeighbors()[0]
            modification[neighbor.GetIntProp("_idx")].append(str(atom.GetSymbol()))
            atom.SetAtomicNum(1)

        if atom.GetSymbol() == "O":
            # C=O -> C
            if len(atom.GetNeighbors()) == 1:
                neighbor = atom.GetNeighbors()[0]
                if neighbor.GetSymbol() == "C":
                    if fragment.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        modification[neighbor.GetIntProp("_idx")].append("=O")
                        to_delete.append(atom.GetIdx())                    
                
            # -O- -> -C-
            else:
                modification[atom.GetIntProp("_idx")].append("O")
                atom.SetAtomicNum(6)
            
        # replace S with O if S has fewer than 2 bonds
        if atom.GetSymbol() == "S":
            # S=O -> C
            if len(atom.GetNeighbors()) == 1:
                neighbor = atom.GetNeighbors()[0]
                if neighbor.GetSymbol() == "C":
                    if fragment.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        modification[neighbor.GetIntProp("_idx")].append("=S")
                        to_delete.append(atom.GetIdx())                    
                
            # -O- -> -C-
            elif len(atom.GetNeighbors()) == 2:
                modification[atom.GetIntProp("_idx")].append("S")
                atom.SetAtomicNum(6)
                
            elif len(atom.GetNeighbors()) == 3:
                if sum([neighbor.GetAtomicNum() == 8 for neighbor in atom.GetNeighbors()]) == 1:
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetAtomicNum() == 8:
                            to_delete.append(neighbor.GetIdx())
                    modification[atom.GetIntProp("_idx")].append("SO")
                    atom.SetAtomicNum(6)
                
            elif len(atom.GetNeighbors()) == 4:
                if sum([neighbor.GetAtomicNum() == 8 for neighbor in atom.GetNeighbors()]) == 2:
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetAtomicNum() == 8:
                            to_delete.append(neighbor.GetIdx())
                    modification[atom.GetIntProp("_idx")].append("SO2")
                    atom.SetAtomicNum(6)

        if atom.GetSymbol() == "N":
            # if explicit valence is less than 3, change to Carbon
            if atom.GetValence(Chem.EXPLICIT) <= 3 and atom.GetFormalCharge() == 0:
                atom.SetAtomicNum(6)  # Change to Carbon
                modification[atom.GetIntProp("_idx")].append("N")
            
            if atom.GetFormalCharge() > 0:
                if not any([neighbor.GetAtomicNum() == 8 for neighbor in atom.GetNeighbors()]):
                    charge = atom.GetFormalCharge()
                    atom.SetFormalCharge(0)
                    atom.SetAtomicNum(6)
                    modification[atom.GetIntProp("_idx")].append(f"N+{charge}")
                
    for idx in sorted(list(set(to_delete)), reverse=True):
        fragment.RemoveAtom(idx)
    
    Chem.SanitizeMol(fragment)    
    fragment = fragment.GetMol()
        
    # reconstruct and canonicalize the fragment
    canonicalize_fragment = Chem.MolFromSmiles(
        Chem.MolToSmiles(Chem.RemoveHs(fragment), canonical=True)
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

def uncanonicalize_fragment(canonical, modification):
    """ Revert a canonicalized fragment to its original form.
    
    Parameters
    ----------
    canonical: rdkit.Chem.Mol
        The canonicalized fragment.
        
    modification: dict
        The mapping from the canonical index to the original atom number.
        
    Returns
    -------
    fragment: rdkit.Chem.Mol
        The original fragment.
    """
    fragment = Chem.RWMol(canonical)
    for idx, mods in modification.items():
        atom = fragment.GetAtomWithIdx(idx)
        for mod in mods:
            if mod == "O":
                atom.SetAtomicNum(8)
            elif mod == "N":
                atom.SetAtomicNum(7)
            elif mod.startswith("N+"):
                charge = int(mod[2:]) if len(mod) > 2 else 1
                atom.SetAtomicNum(7)
                atom.SetFormalCharge(charge)
            elif mod == "S":
                atom.SetAtomicNum(16)
            elif mod.startswith("SO"):
                atom.SetAtomicNum(16)
                oxygen_count = int(mod[2:]) if len(mod) > 2 else 1
                for _ in range(oxygen_count):
                    new_atom = Chem.Atom(8)
                    new_idx = fragment.AddAtom(new_atom)
                    fragment.AddBond(atom.GetIdx(), new_idx, order=Chem.rdchem.BondType.SINGLE)
            elif mod.startswith("="):
                new_atom = Chem.Atom(mod[1:])
                new_idx = fragment.AddAtom(new_atom)
                fragment.AddBond(atom.GetIdx(), new_idx, order=Chem.rdchem.BondType.DOUBLE)
            elif mod in ["F", "Cl", "Br", "I"]:
                new_atom = Chem.Atom(mod)
                new_idx = fragment.AddAtom(new_atom)
                fragment.AddBond(atom.GetIdx(), new_idx, order=Chem.rdchem.BondType.SINGLE)
            else:
                raise ValueError(f"Unknown modification: {mod}")
    Chem.SanitizeMol(fragment)
    return fragment.GetMol()
    


_NON_RING_SINGLE = Chem.MolFromSmarts("[*]-&!@[*]")
def get_rotatable_bonds(molecule):
    return molecule.GetSubstructMatches(_NON_RING_SINGLE)

_EXOCYCLIC = Chem.MolFromSmarts("[R]!@[!R]")
def get_exocyclic_bonds(molecule):
    return molecule.GetSubstructMatches(_EXOCYCLIC)

def break_fused_ring(fragment):
    """ Break fused rings in a fragment by cutting one bond in each fused ring.
    
    Parameters
    ----------
    fragment: rdkit.Chem.Mol
        The fragment to be processed.
        
    Returns
    -------
    fragments: list of rdkit.Chem.Mol
        The list of fragments after breaking fused rings.
        
    Examples
    --------
    >>> fragment = Chem.MolFromSmiles("C1CCC2CCCCC2C1")
    >>> fragments = break_fused_ring(fragment)
    >>> print([Chem.MolToSmiles(frag) for frag in fragments])
    """
    print("Breaking fused rings for fragment:", Chem.MolToSmiles(fragment))
    for idx in range(len(fragment.GetAtoms())):
        atom = fragment.GetAtomWithIdx(idx)
        atom.SetIntProp("_idx", atom.GetIdx())


    Chem.SanitizeMol(fragment)
    ring_info = fragment.GetRingInfo()
    
    # let it go if there is no ring
    if ring_info.NumRings() <= 1:
        return [fragment]
    

    
    fragment = Chem.RWMol(fragment)
    def num_shared_atoms(ring1, ring2):
        return len(set(ring1) & set(ring2))
    
    # enumerate atom rings
    all_sharing_three = True
    rings = list(ring_info.AtomRings())
    for idx, ring in enumerate(rings):
        other_rings = [r for r in rings if r != ring]
        
        # determine if the ring is safe to remove
        if all([num_shared_atoms(ring, other) <= 2 for other in other_rings]):
            all_sharing_three = False
            # build a new fragment that is exclusively the ring
            new_fragment = Chem.RWMol()
            mapping = {}
            for atom in ring:
                atom = fragment.GetAtomWithIdx(atom)
                new_atom = Chem.Atom(atom.GetAtomicNum())
                idx = new_fragment.AddAtom(new_atom)
                mapping[atom.GetIntProp("_idx")] = idx
                
            for atom0 in ring:
                for atom1 in ring:
                    if atom0 < atom1:
                        bond = fragment.GetBondBetweenAtoms(atom0, atom1)
                        if bond is not None:
                            begin = bond.GetBeginAtom().GetIntProp("_idx")
                            end = bond.GetEndAtom().GetIntProp("_idx")
                            new_fragment.AddBond(mapping[begin], mapping[end], bond.GetBondType())
                            
            # remove the ring from the original fragment
            to_remove = []
            for atom in sorted(ring):
                if not any([atom in other for other in other_rings]):
                    to_remove.append(atom)
            to_remove = sorted(to_remove, reverse=True)
            for atom in to_remove:
                fragment.RemoveAtom(atom)
                
            result = [new_fragment.GetMol()] + break_fused_ring(fragment.GetMol())
            break
    
    if all_sharing_three:
        result = [fragment.GetMol()]

    return result
        
            

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
    molecule = Chem.RemoveHs(molecule)
    Chem.RemoveStereochemistry(molecule)
    Chem.Kekulize(molecule)
    print("original molecule:", Chem.MolToSmiles(molecule))
    
    for atom in molecule.GetAtoms():
        atom.SetIntProp("_idx", atom.GetIdx())
    rotatable_bonds = get_rotatable_bonds(molecule) + get_exocyclic_bonds(molecule)
    rotatable_bonds = set([tuple(sorted(bond)) for bond in rotatable_bonds])
    
    rotatable_bond_idxs = [molecule.GetBondBetweenAtoms(*bond) for bond in rotatable_bonds]
    if len(rotatable_bond_idxs) == 0:
        fragments = [molecule]
    else:
        fragments = Chem.FragmentOnBonds(molecule, [bond.GetIdx() for bond in rotatable_bond_idxs], addDummies=False)
        fragments = Chem.GetMolFrags(fragments, asMols=True, sanitizeFrags=False)            
    # fragments, renumbering, modifications = zip(*[canonicalize_fragment(frag) for frag in fragments])
    fragments = [break_fused_ring(frag) for frag in fragments]
    fragments = [frag for sublist in fragments for frag in sublist]
    fragments = [Chem.MolToSmiles(frag, canonical=True) for frag in fragments]
    # return fragments, renumbering, modifications, rotatable_bonds
    return fragments
    
def fragments_to_tree(fragments, renumbering, modifications, rotatable_bonds):
    """ Build a tree from the fragments of a molecule. """
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
    tree = fragments_to_tree(fragments, renumbering, modifications, rotatable_bonds)
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
    
def build_library(molecules):
    library = []
    for smiles in tqdm.tqdm(molecules):
        fragments = generate_fragments(smiles)
        library += fragments
    # count occurrences
    from collections import Counter
    for fragment, count in Counter(library).most_common(len(Counter(library))):
        print(f"{fragment}\t{count}")
    library = set(library)
    return library
    
        
        
        
        