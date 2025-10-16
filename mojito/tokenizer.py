from collections import defaultdict
from functools import reduce  
import token
import zlib
from typing import Sequence, Optional
from rdkit import Chem
from rdkit import RDLogger
import networkx as nx
import tqdm
from rdkit.Chem import Draw
RDLogger.DisableLog('rdApp.*') 

def smiles(fn):
    def wrapper(*args, **kwargs):
        if isinstance(args[0], str):
            args = (Chem.MolFromSmiles(args[0]),) + args[1:]
        return fn(*args, **kwargs)
    wrapper.__doc__ = fn.__doc__
    return wrapper

_hash = lambda fragment: zlib.adler32(
    Chem.MolToSmiles(fragment, canonical=True).encode("utf-8")
) if isinstance(fragment, Chem.Mol) else zlib.adler32(
    fragment.encode("utf-8")
)

_NON_RING_SINGLE = Chem.MolFromSmarts("[*]-&!@[*]")
def get_rotatable_bonds(molecule):
    return molecule.GetSubstructMatches(_NON_RING_SINGLE)

_EXOCYCLIC = Chem.MolFromSmarts("[R]~!@[*]")
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
    
    # if three rings share atoms, merge them
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            for k in range(j + 1, len(rings)):
                if (
                    num_shared_atoms(rings[i], rings[j]) > 0 and
                    num_shared_atoms(rings[j], rings[k]) > 0 and
                    num_shared_atoms(rings[i], rings[k]) > 0
                ):
                    rings[i] = list(set(rings[i]) | set(rings[j]) | set(rings[k]))
                    rings[j] = []
                    rings[k] = []
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
    Chem.Kekulize(fragment, clearAromaticFlags=True)
    canonical = Chem.MolToSmiles(fragment, canonical=True)    
    canonical = Chem.MolFromSmiles(canonical)
    Chem.Kekulize(canonical, clearAromaticFlags=True)
    
    for atom in canonical.GetAtoms():
        atom.SetNumRadicalElectrons(0)
    
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
        

def _molecule_to_fragments(molecule):
    # break by bonds
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
    
    return fragments, rotatable_bonds

@smiles
def molecule_to_fragments(molecule):
    molecule = Chem.RemoveHs(molecule)
    Chem.RemoveStereochemistry(molecule)
    Chem.Kekulize(molecule, clearAromaticFlags=True)
    for atom in molecule.GetAtoms():
        atom.SetIntProp("_idx", atom.GetIdx())
    fragments, _ = _molecule_to_fragments(molecule)
    # return smiles
    fragments = [Chem.MolToSmiles(frag, canonical=True) for frag in fragments]
    return fragments
        
@smiles
def molecule_to_tree(molecule, score=_hash):
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

    # call the internal function to get fragments and rotatable bonds
    fragments, rotatable_bonds = _molecule_to_fragments(molecule)
    
    # rank fragments by score
    fragments = sorted(fragments, key=score, reverse=True)
        
    # build a tree
    tree = nx.DiGraph()
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
        src_global, src_local = atom_to_fragment[old_src][0]
        dst_global, dst_local = atom_to_fragment[old_dst][0]
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
                
                
    # assert connected
    assert nx.is_connected(tree.to_undirected())
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
    molecule = Chem.RWMol(molecule)
    
    def find_atom(molecule, global_idx, local_idx):
        for atom in molecule.GetAtoms():
            if atom.GetIntProp("_global_idx") == global_idx and atom.GetIntProp("_local_idx") == local_idx:
                return atom
        return None
    
    # loop through edges to add bonds
    for src_global, dst_global, data in tree.edges(data=True):
        src_local = data["src_local"]
        dst_local = data["dst_local"]
        bond_type = data["bond_type"]
        if bond_type > 0:
            src_atom = find_atom(molecule, src_global, src_local)
            dst_atom = find_atom(molecule, dst_global, dst_local)
            assert src_atom is not None and dst_atom is not None
            molecule.AddBond(src_atom.GetIdx(), dst_atom.GetIdx(), order=Chem.rdchem.BondType(bond_type))
        
    # build fused rings
    overlapping = nx.Graph()
    for src_global, dst_global, data in tree.edges(data=True):
        if data["bond_type"] == 0:
            src_local = data["src_local"]
            dst_local = data["dst_local"]
            src_atoms = [find_atom(molecule, src_global, idx) for idx in src_local]
            dst_atoms = [find_atom(molecule, dst_global, idx) for idx in dst_local]
            
            for src_atom, dst_atom in zip(src_atoms, dst_atoms):
                overlapping.add_edge(src_atom.GetIdx(), dst_atom.GetIdx())
            
    to_remove = []        
    for component in nx.connected_components(overlapping):
        component = [int(idx) for idx in component]
        component = sorted(component)
        base = component[0]
        for idx in component[1:]:
            for neighbor in molecule.GetAtomWithIdx(idx).GetNeighbors():
                neighbor = int(neighbor.GetIdx())
                old_bond = molecule.GetBondBetweenAtoms(idx, neighbor)
                if old_bond is not None:
                    if molecule.GetBondBetweenAtoms(base, neighbor) is None:
                        molecule.AddBond(base, neighbor, order=old_bond.GetBondType())
            to_remove.append(idx)

    for idx in sorted(set(to_remove), reverse=True):
        molecule.RemoveAtom(idx)
                
    molecule = molecule.GetMol()
    
    def can_sanitize(molecule):
        try:
            Chem.SanitizeMol(molecule)
            return True
        except:
            return False
        
    if not can_sanitize(molecule):
        # make editable        
        molecule = Chem.AddHs(molecule)
        molecule = Chem.RWMol(molecule)
        
        for atom in molecule.GetAtoms():
            atom.SetNoImplicit(True)

        
        to_remove = []
        for atom in molecule.GetAtoms():
            valence = atom.GetTotalValence()
            permitted = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
            charge = min(atom.GetFormalCharge(), 0)
        
            if valence - charge > permitted:
                num_excess = valence - charge - permitted
                hydrogen_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == "H"]
                to_remove.extend([h.GetIdx() for h in hydrogen_neighbors[:num_excess]])
        
        while not can_sanitize(molecule) and len(to_remove) > 0:
            idx = to_remove.pop()
            molecule.RemoveAtom(idx)
            
    # if there is radicals, attach or delete hydrogens
    num_atoms = molecule.GetNumAtoms()
    if any(atom.GetNumRadicalElectrons() > 0 for atom in molecule.GetAtoms()):
        molecule = Chem.AddHs(molecule)
        molecule = Chem.RWMol(molecule)
    
        to_remove = []
        for idx in range(num_atoms):
            atom = molecule.GetAtomWithIdx(idx)
                
            if atom.GetNumRadicalElectrons()==1 and atom.GetFormalCharge()==1:
                h = Chem.Atom(1)
                h.SetNoImplicit(True)
                h_idx = molecule.AddAtom(h)
                molecule.AddBond(idx, h_idx, order=Chem.rdchem.BondType.SINGLE)
                
            if atom.GetNumRadicalElectrons()==1 and atom.GetFormalCharge()==-1:
                atom.SetNoImplicit(True)
                hydrogen_neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == "H"]
                to_remove.append(hydrogen_neighbors[0])
        
        for idx in sorted(set(to_remove), reverse=True):
            molecule.RemoveAtom(idx)
                
        molecule = molecule.GetMol()
        Chem.SanitizeMol(molecule) 
        
    molecule = Chem.RemoveHs(molecule)
    return molecule

def tree_to_tokens(tree, score=_hash):
    ordered_edges = [(u, v) for u, v in tree.edges()]
    # tree = tree.to_directed()
    tree = nx.Graph(tree)  # make it undirected
    tokens = []
    fragments = [data["fragment"] for _, data in tree.nodes(data=True)]
    scores = [score(Chem.MolFromSmiles(frag)) for frag in fragments]
    for idx, s in zip(tree.nodes(), scores):
        tree.nodes[idx]["score"] = s
    source = max(tree.nodes, key=lambda idx: tree.nodes[idx]["score"])
    
    
    tokens.append(tree.nodes[source]["fragment"])
    dfs = nx.dfs_labeled_edges(
        tree, 
        source=source,
        sort_neighbors=lambda neighbors: sorted(neighbors, key=lambda idx: tree.nodes[idx]["score"], reverse=True),
    )

    for src, dst, direction in dfs:
        if src == dst:
            continue
        
        if (dst, src) in ordered_edges:
            assert (src, dst) not in ordered_edges
            # dst, src = src, dst
            flipped = True
        else:
            flipped = False
        
        if direction == "reverse":
            if "BACK" in tokens[-1]:
                number_of_backs = int(tokens[-1][5:]) + 1
                tokens[-1] = f"BACK {number_of_backs}"
            else:
                tokens.append("BACK 1")
                
        elif direction == "forward":                
            edge = tree.get_edge_data(src, dst)
            edge_token = "EDGE "
            
            # handle the source
            src_local = edge["src_local"] if not flipped else edge["dst_local"]
            edge_token += str(src_local)

            # handle the bond type string
            bond_type = edge["bond_type"]
            edge_token += {
                0.0: ":",
                1.0: "-",
                2.0: "=",
                3.0: "#",
            }[bond_type]
            
            # handle the destination
            dst_local = edge["dst_local"] if not flipped else edge["src_local"]
            edge_token += str(dst_local)
                
            if edge_token != "EDGE 0-0":
                tokens.append(edge_token)
            
            tokens.append(tree.nodes[dst]["fragment"])
            
    is_carbon = lambda token: set(token) == {"C"}
    cursor = 0
    while cursor < len(tokens) - 1:
        while is_carbon(tokens[cursor]) and is_carbon(tokens[cursor + 1]):
            tokens[cursor] = tokens[cursor] + tokens.pop(cursor + 1)
        cursor += 1
                        
    tokens = [f"<{token}>" for token in tokens]
    while "BACK" in tokens[-1]:
        tokens.pop()
    return tokens
    
def tokens_to_tree(tokens):
    # define token types
    is_edge = lambda token: "EDGE" in token
    is_back = lambda token: "BACK" in token
    is_fragment = lambda token: not (is_edge(token) or is_back(token))
    
    # replace carbon chain tokens with multiple C's
    is_carbon = lambda token: set(token) == {"C"}
    cursor = 0
    while cursor < len(tokens):
        token = tokens[cursor]
        if is_carbon(token[1:-1]) and len(token) > 3:
            num_carbons = len(token) - 2
            tokens = tokens[:cursor] + ["<C>"] * num_carbons + tokens[cursor+1:]
        cursor += 1
        
    # insert missing <EDGE 0-0> tokens
    cursor = 0
    while cursor < len(tokens) - 1:
        if not is_edge(tokens[cursor]) and is_fragment(tokens[cursor + 1]):
            tokens.insert(cursor + 1, "<EDGE 0-0>")
        cursor += 1
    
    tree = nx.Graph()
    edge_to_add = None
    layers = defaultdict(list)
    cursor = 0
    while tokens:
        token = tokens.pop(0)
        if is_back(token):
            number_of_backs = int(token[5:-1])
            cursor -= number_of_backs
        
        elif is_fragment(token):
            idx = len(tree)
            tree.add_node(
                idx,
                fragment=token[1:-1],
            )
            if edge_to_add is not None:
                tree.add_edge(
                    layers[cursor][-1],
                    idx,
                    **edge_to_add,
                )

            cursor += 1
            layers[cursor].append(idx)

        elif is_edge(token):
            edge_token = token[6:-1]
            bond_type_char = next(c for c in edge_token if not c.isdigit() and c not in " [],")
            bond_type = {
                ":": 0.0,
                "-": 1.0,
                "=": 2.0,
                "#": 3.0,
            }[bond_type_char]
            src_local, dst_local = edge_token.split(bond_type_char)
            src_local, dst_local = eval(src_local), eval(dst_local)
            edge_to_add = {
                "src_local": src_local,
                "dst_local": dst_local,
                "bond_type": bond_type,
            }
            

            

    return tree
    
    
def molecule_to_tokens(molecule):
    return tree_to_tokens(molecule_to_tree(molecule))

def tokens_to_molecule(tokens):
    return tree_to_molecule(tokens_to_tree(tokens))
    
def build_library(molecules):
    library = []
    for smiles in tqdm.tqdm(molecules):
        try:
            old_molecule = Chem.MolFromSmiles(smiles)
            Chem.RemoveStereochemistry(old_molecule)
            Chem.Kekulize(old_molecule, clearAromaticFlags=True)
            fragments = molecule_to_fragments(old_molecule)
            library.extend(fragments)
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
        
    # count unique fragments
    from collections import Counter
    counter = dict(Counter(library))
    return counter
    
        
        
        
        