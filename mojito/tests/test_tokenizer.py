import torch

from mojito import FSQ, Decoder, Encoder, Tokenizer, reconstruction_loss
from mojito.chem import canonical, parse, to_mol
from mojito.data import build_vocab, collate, featurize, split
from mojito.metrics import Accumulator, batch_counts, collisions

SMILES = [
    "CCO",
    "c1ccccc1O",
    "CC(=O)Nc1ccc(O)cc1",
    "C[NH3+]",
    "O=C([O-])c1ccc[nH]1",
    "C#CC1CC1",
]


def _tokenizer(num_atom_types, levels=(8, 5, 5, 5)):
    fsq = FSQ(levels)
    return Tokenizer(
        Encoder(num_atom_types, 32, fsq.dim, depth=2, num_heads=4),
        fsq,
        Decoder(fsq.dim, 32, num_atom_types, depth=2, num_heads=4),
    )


def test_graph_round_trip():
    keys = build_vocab(SMILES)
    graphs, dropped = featurize(SMILES, keys)
    assert dropped == 0
    for smiles, g in zip(SMILES, graphs):
        mol = to_mol(g.atoms, g.bonds, keys)
        assert canonical(mol) == canonical(parse(smiles))


def test_split_is_disjoint_and_deduplicated():
    smiles = SMILES + ["OCC", "Oc1ccccc1"]  # duplicates of 0 and 1
    parts = split(smiles, seed=0)
    flat = [s for p in parts for s in p]
    assert len(flat) == len(set(flat)) == len(SMILES)


def test_fsq_index_round_trip():
    fsq = FSQ([8, 5, 5, 5])
    codes = fsq(torch.randn(100, 4) * 3)
    assert torch.equal(fsq.to_codes(fsq.to_indices(codes)), codes)
    assert fsq.to_indices(codes).max() < fsq.codebook_size


def test_forward_shapes_and_padding_invariance():
    torch.manual_seed(0)
    keys = build_vocab(SMILES)
    graphs, _ = featurize(SMILES, keys)
    model = _tokenizer(len(keys)).eval()
    batch = collate(graphs)
    atom_logits, bond_logits, indices = model(batch)
    b, n = batch.mask.shape
    assert atom_logits.shape == (b, n, len(keys))
    assert bond_logits.shape == (b, n, n, 4)
    assert torch.allclose(bond_logits, bond_logits.transpose(1, 2))
    # a molecule's tokens must not depend on its batch-mates
    alone = model.encode(collate(graphs[:1]))
    m = len(graphs[0].atoms)
    assert torch.equal(alone[0, :m], indices[0, :m])
    loss_atom, loss_bond = reconstruction_loss(
        atom_logits, bond_logits, batch
    )
    (loss_atom + loss_bond).backward()


def test_metrics_on_perfect_prediction():
    keys = build_vocab(SMILES)
    batch = collate(featurize(SMILES, keys)[0])
    atom_logits = torch.nn.functional.one_hot(batch.atoms, len(keys))
    bond_logits = torch.nn.functional.one_hot(batch.bonds, 4)
    indices = torch.arange(batch.atoms.numel()).view_as(batch.atoms)
    acc = Accumulator()
    acc.update(batch_counts(atom_logits, bond_logits, indices, batch))
    out = acc.summarize(codebook_size=1000)
    assert out["exact"] == out["bond_f1"] == out["atom_acc"] == 1.0
    assert out["exact_upper_bound"] == 1.0


def test_collision_detection():
    keys = build_vocab(["CCO"])
    batch = collate(featurize(["CCO"], keys)[0])
    # C(methyl), C(methylene), O are mutually non-equivalent
    assert collisions(torch.tensor([[1, 1, 2]]), batch.symmetry, batch.mask)
    assert not collisions(
        torch.tensor([[1, 2, 3]]), batch.symmetry, batch.mask
    )
    # ethane's two carbons are equivalent: shared token is fine
    batch = collate(featurize(["CC"], keys)[0])
    assert not collisions(torch.tensor([[7, 7]]), batch.symmetry, batch.mask)
