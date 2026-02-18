import torch
from ._like import likeable

#### Pauli ####
@likeable
def ID1_like(*, dtype, device):
    return torch.eye(2, dtype=dtype, device=device)

@likeable
def sigma_Z_like(*, dtype, device):
    return torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)

@likeable
def sigma_X_like(*, dtype, device):
    return torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)

@likeable
def sigma_Y_like(*, dtype, device):
    return torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)

#### Projections ####
@likeable
def ketbra00_like(*, dtype, device):
    return torch.tensor([[1, 0], [0, 0]], dtype=dtype, device=device)

@likeable
def ketbra01_like(*, dtype, device):
    return torch.tensor([[0, 1], [0, 0]], dtype=dtype, device=device)

@likeable
def ketbra10_like(*, dtype, device):
    return torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device)

@likeable
def ketbra11_like(*, dtype, device):
    return torch.tensor([[0, 0], [0, 1]], dtype=dtype, device=device)
