from dataclasses import dataclass
from collections import defaultdict, namedtuple
from functools import reduce  
from typing import Sequence, Optional
from rdkit import Chem
import networkx as nx
import tqdm
from rdkit.Chem import Draw
import os

def smiles(fn):
    def wrapper(*args, **kwargs):
        if isinstance(args[0], str):
            args = (Chem.MolFromSmiles(args[0]),) + args[1:]
        return fn(*args, **kwargs)
    wrapper.__doc__ = fn.__doc__
    return wrapper

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
    >>> from rdkit import Chem
    >>> frag = Chem.MolFromSmiles("C1CCC2CCCCC2C1")
    >>> frags = break_fused_ring(frag)
    
    """
    rings = [list(ring) for ring in Chem.GetSymmSSSR(fragment)]
    
    if len(rings) <= 1:
        return [fragment]
    
    def num_shared_atoms(ring1, ring2):
        return len(set(ring1) & set(ring2))
    
    # merge rings that share 3 or more atoms
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if num_shared_atoms(rings[i], rings[j]) >= 3:
                rings[i] = list(set(rings[i]) | set(rings[j]))
                rings[j] = []
    rings = [ring for ring in rings if len(ring) > 0]
    
    def subgraph(fragment, ring):
        new_fragment = Chem.RWMol()
        mapping = {}
        for atom in ring:
            atom = fragment.GetAtomWithIdx(atom)
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetIntProp("_idx", atom.GetIntProp("_idx"))
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            idx = new_fragment.AddAtom(new_atom)
            mapping[atom.GetIntProp("_idx")] = idx
            
        for atom0 in ring:
            for atom1 in ring:
                if atom0 < atom1:
                    bond = fragment.GetBondBetweenAtoms(atom0, atom1)
                    if bond is not None:
                        begin = bond.GetBeginAtom().GetIntProp("_idx")
                        end = bond.GetEndAtom().GetIntProp("_idx")
                        if new_fragment.GetBondBetweenAtoms(mapping[begin], mapping[end]) is None:
                            new_fragment.AddBond(mapping[begin], mapping[end], bond.GetBondType())
        fragment = new_fragment.GetMol()
        Chem.Kekulize(fragment)
        return fragment
    
    rings = [subgraph(fragment, ring) for ring in rings]
    return rings
            
def canonicalize_fragment(fragment):
    """ Canonicalize a fragment and return the renumbering.
    
    Parameters
    ----------
    fragment: rdkit.Chem.Mol
        The fragment to be canonicalized.
        
    Returns
    -------
    canonical_fragment: rdkit.Chem.Mol
        The canonicalized fragment.
        
    renumbering: Bijection
        The mapping from the original fragment to the canonicalized fragment.
        
    modifications: dict
        The mapping from the canonical index to the original atom number.
        
    """
    canonical = Chem.MolToSmiles(fragment, canonical=True)
    canonical = Chem.MolFromSmiles(canonical)
    Chem.Kekulize(canonical, clearAromaticFlags=True)
    Chem.Kekulize(fragment, clearAromaticFlags=True)
    
    if canonical.GetNumAtoms() == 1:
        # single atom fragment
        canonical.GetAtomWithIdx(0).SetIntProp(
            "_idx", fragment.GetAtomWithIdx(0).GetIntProp("_idx")
        )
        return canonical
    
    renumbering = fragment.GetSubstructMatch(canonical, useChirality=False)
    for new, old in enumerate(renumbering):
        canonical.GetAtomWithIdx(new).SetIntProp(
            "_idx", fragment.GetAtomWithIdx(old).GetIntProp("_idx")
        )

    for atom in canonical.GetAtoms():
        try:
            atom.GetIntProp("_idx")
        except KeyError:
            print(Chem.MolToSmiles(fragment, canonical=True), Chem.MolToSmiles(fragment, canonical=False), renumbering)
            raise ValueError("Canonicalization failed.")
    return canonical
        
@smiles
def molecule_to_tree(molecule):
    """ Generate fragments by cutting rotatable bonds.
    
    Parameters
    ----------
    molecule: rdkit.Chem.Mol
        The molecule to be fragmented.
                
    """
    molecule = Chem.RemoveHs(molecule)
    Chem.RemoveStereochemistry(molecule)
    Chem.Kekulize(molecule, clearAromaticFlags=True)
    
    # break by bonds
    for atom in molecule.GetAtoms():
        atom.SetIntProp("_idx", atom.GetIdx())
    rotatable_bonds = get_rotatable_bonds(molecule) + get_exocyclic_bonds(molecule)
    rotatable_bonds = set([tuple(sorted(bond)) for bond in rotatable_bonds])
    rotatable_bond_idxs = [molecule.GetBondBetweenAtoms(*bond) for bond in rotatable_bonds]
        
    if len(rotatable_bond_idxs) > 0:
        fragments = Chem.FragmentOnBonds(molecule, [bond.GetIdx() for bond in rotatable_bond_idxs], addDummies=False)
        fragments = Chem.GetMolFrags(fragments, asMols=True, sanitizeFrags=False)    
    else:
        fragments = [molecule]
        
    
    # break by fused rings
    fragments = [break_fused_ring(frag) for frag in fragments]
    fragments = [frag for sublist in fragments for frag in sublist]    
    
    # canonicalize fragments
    fragments = [canonicalize_fragment(frag) for frag in fragments]
        
    # build a tree
    tree = nx.Graph()
    for idx, fragment in enumerate(fragments):
        tree.add_node(
            idx,
            fragment=Chem.MolToSmiles(fragment, canonical=True),
        )
        
    # build mapping from old atom idx to (fragment idx, local atom idx)
    atom_to_fragment = defaultdict(list)
    for idx, fragment in enumerate(fragments):
        for atom in fragment.GetAtoms():
            atom_to_fragment[atom.GetIntProp("_idx")].append((idx, atom.GetIdx()))
        
    # add rotatable and exocyclic bonds as edges
    for old_src, old_dst in rotatable_bonds:
        bond_type = molecule.GetBondBetweenAtoms(old_src, old_dst).GetBondTypeAsDouble()
        for src_global, src_local in atom_to_fragment[old_src]:
            for dst_global, dst_local in atom_to_fragment[old_dst]:
                    tree.add_edge(
                        src_global,
                        dst_global,
                        src_local=src_local,
                        dst_local=dst_local,
                        bond_type=bond_type,
                    )
                    
    # add edges among fused rings
    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            shared_atoms = set(
                atom.GetIntProp("_idx") for atom in fragments[i].GetAtoms()
            ) & set(
                atom.GetIntProp("_idx") for atom in fragments[j].GetAtoms()
            )

            if len(shared_atoms) > 0:
                bond_type = 0.0
                src_global = i
                dst_global = j
                src_local = []
                dst_local = []
                for atom in shared_atoms:
                    for atom_i in fragments[i].GetAtoms():
                        if atom_i.GetIntProp("_idx") == atom:
                            src_local.append(atom_i.GetIdx())
                    for atom_j in fragments[j].GetAtoms():
                        if atom_j.GetIntProp("_idx") == atom:
                            dst_local.append(atom_j.GetIdx())
                

                tree.add_edge(
                    src_global,
                    dst_global,
                    src_local=src_local,
                    dst_local=dst_local,
                    bond_type=bond_type,
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
            
    
    """
    # record the origin of each atom in the final molecule
    fragments = []
    for idx, data in tree.nodes(data=True):
        fragment = Chem.MolFromSmiles(data["fragment"])
        Chem.Kekulize(fragment, clearAromaticFlags=True)
        
        # modification = data["modification"]
        for atom in fragment.GetAtoms():
            atom.SetIntProp("_global_idx", idx)
            atom.SetIntProp("_local_idx", atom.GetIdx())
        fragments.append(fragment)
                
    # combine the fragment into one molecule
    molecule = reduce(Chem.CombineMols, fragments)
    molecule = Chem.EditableMol(molecule)
    
    def find_atom(molecule, global_idx, local_idx):
        for atom in molecule.GetMol().GetAtoms():
            if atom.GetIntProp("_global_idx") == global_idx and atom.GetIntProp("_local_idx") == local_idx:
                return atom
        return None
    
    # loop through edges to add bonds
    to_delete = []
    for src_global, dst_global, data in tree.edges(data=True):
        src_local = data["src_local"]
        dst_local = data["dst_local"]
        bond_type = data["bond_type"]
        if bond_type > 0:
            src_atom = find_atom(molecule, src_global, src_local)
            dst_atom = find_atom(molecule, dst_global, dst_local)
            assert src_atom is not None and dst_atom is not None
            molecule.AddBond(src_atom.GetIdx(), dst_atom.GetIdx(), order=Chem.rdchem.BondType(bond_type))
        
        else:
            src_atoms = [find_atom(molecule, src_global, idx) for idx in src_local]
            dst_atoms = [find_atom(molecule, dst_global, idx) for idx in dst_local]
            
            for src_atom, dst_atom in zip(src_atoms, dst_atoms):
                src_idx, dst_idx = src_atom.GetIdx(), dst_atom.GetIdx()
                
                # ensure src_idx < dst_idx
                if src_idx > dst_idx:
                    src_idx, dst_idx = dst_idx, src_idx
                    src_atom, dst_atom = dst_atom, src_atom
                    
                for neighbors in dst_atom.GetNeighbors():
                    old_bond = molecule.GetMol().GetBondBetweenAtoms(dst_idx, neighbors.GetIdx())
                    if molecule.GetMol().GetBondBetweenAtoms(src_idx, neighbors.GetIdx()) is None:
                        molecule.AddBond(src_idx, neighbors.GetIdx(), order=old_bond.GetBondType())
                
                to_delete.append(dst_idx)
                
    
    for idx in sorted(set(to_delete), reverse=True):
        molecule.RemoveAtom(idx)            
            
    molecule = molecule.GetMol()

    
    try:    
        Chem.SanitizeMol(molecule)
    except:        
        # add hydrogens
        molecule = Chem.AddHs(molecule)
    
        # make editable        
        molecule = Chem.RWMol(molecule)
        
        
        to_remove = []
        for atom in molecule.GetAtoms():
            valence = atom.GetTotalValence()
            permitted = max(Chem.GetPeriodicTable().GetValenceList(atom.GetAtomicNum()))
            charge = min(atom.GetFormalCharge(), 0)
            
            if valence - charge > permitted:
                num_extra_hydrogens = valence - charge - permitted
                
                # if atom is sp2, remove one fewer hydrogen
                if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                    num_extra_hydrogens -= 1
                
                atom.SetNoImplicit(True)
                hydrogen_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == "H"]
                to_remove.extend([h.GetIdx() for h in hydrogen_neighbors[:num_extra_hydrogens]])
                atom.SetNoImplicit(False)

        # remove in reverse order to preserve indices
        for idx in sorted(set(to_remove), reverse=True):
            molecule.RemoveAtom(idx)

        Chem.SanitizeMol(molecule)
        molecule = molecule.GetMol()
        molecule = Chem.RemoveHs(molecule)
                    
    return molecule
    
def build_library(molecules):
    library = []
    for smiles in tqdm.tqdm(molecules):
        old_molecule = Chem.MolFromSmiles(smiles)
        Chem.RemoveStereochemistry(old_molecule)
        Chem.Kekulize(old_molecule, clearAromaticFlags=True)
        old_smiles = Chem.MolToSmiles(old_molecule, canonical=True)
        
        tree = molecule_to_tree(smiles)
        new_molecule = tree_to_molecule(tree)
        new_smiles = Chem.MolToSmiles(new_molecule, canonical=True)
        
        if old_smiles != new_smiles:
            print(f"Warning: {old_smiles} != {new_smiles}")
            # raise ValueError("Molecule reconstruction failed.")
        
        
        fragments = [data["fragment"] for _, data in tree.nodes(data=True)]
        library.extend(fragments)
        
    library = set(library)
    return library
    
        
        
        
        