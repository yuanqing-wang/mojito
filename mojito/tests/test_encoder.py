import torch

from mojito import Encoder


def test_permutation_equivariance():
    torch.manual_seed(0)
    n = 5
    atoms = torch.randint(0, 3, (1, n))
    bonds = torch.zeros(1, n, n, dtype=torch.long)
    for i in range(n - 1):  # path graph
        bonds[0, i, i + 1] = bonds[0, i + 1, i] = 1
    idx = torch.arange(n)
    distance = (idx[:, None] - idx[None, :]).abs()[None]
    mask = torch.ones(1, n, dtype=torch.bool)
    encoder = Encoder(3, 16, 4, depth=2, num_heads=2).eval()
    out = encoder(atoms, bonds, distance, mask)
    perm = torch.randperm(n)
    out_perm = encoder(
        atoms[:, perm],
        bonds[:, perm][:, :, perm],
        distance[:, perm][:, :, perm],
        mask,
    )
    assert torch.allclose(out[:, perm], out_perm, atol=1e-5)
