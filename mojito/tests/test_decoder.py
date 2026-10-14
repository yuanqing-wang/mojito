import torch

from mojito import Decoder


def test_padding_does_not_leak():
    torch.manual_seed(0)
    decoder = Decoder(4, 16, 3, depth=2, num_heads=2).eval()
    codes = torch.rand(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    atom, bond = decoder(codes, mask)
    padded = torch.cat([codes, torch.rand(1, 2, 4)], dim=1)
    pmask = torch.tensor([[True] * 3 + [False] * 2])
    atom_p, bond_p = decoder(padded, pmask)
    assert torch.allclose(atom, atom_p[:, :3], atol=1e-5)
    assert torch.allclose(bond, bond_p[:, :3, :3], atol=1e-5)
