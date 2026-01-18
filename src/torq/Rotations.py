import torch
from . import SingleQubitGates as single, Ops as ops


def get_rz(phi, only_first_column=False):
    """
    Compute Rz gate for each angle in the batch.
    angle: tensor of shape [B] (or scalar)
    Returns: tensor of shape [B, 2, 2]
    """
    # Use the device of the input tensor
    ####### Protection #######
    if not torch.isfinite(phi).all():
        bad = (~torch.isfinite(phi)).nonzero(as_tuple=False)[0].tolist()
        raise RuntimeError(f"[nan] get_rz phi @ {bad}")
    ####################
    if phi.dim() == 0:
        phi = phi.unsqueeze(0)
    coeff1 = torch.exp(-1j * (phi / 2)).unsqueeze(-1).unsqueeze(-1)
    zeros_batch = torch.zeros_like(coeff1)
    if only_first_column:
        out = torch.cat([coeff1, zeros_batch], dim=1)
    else:
        coeff2 = torch.exp(1j * (phi / 2)).unsqueeze(-1).unsqueeze(-1)
        # ketbra00 = top_part = torch.tensor([[1, 0], [0, 0]], dtype=dev.dtype_complex, device=device).unsqueeze(0)
        # ketbra11 = bot_part = torch.tensor([[0, 0], [0, 1]], dtype=dev.dtype_complex, device=device).unsqueeze(0)
        out = (coeff1 * single.ketbra00_like(x=coeff1).unsqueeze(0)) + (coeff2 * single.ketbra11_like(x=coeff2).unsqueeze(0))
    ####### Protection #######
    if not torch.isfinite(out).all():
        bad = (~torch.isfinite(out)).nonzero(as_tuple=False)[0].tolist()
        raise RuntimeError(f"[nan] get_rz out @ {bad}")
    ####################
    return out


def get_rx(phi, only_first_column=False):
    """
    Compute Rx gate for each phi in the batch.
    Returns: tensor of shape [B, 2, 2]
    """
    # device = phi.device
    ####### Protection #######
    if not torch.isfinite(phi).all():
        bad = (~torch.isfinite(phi)).nonzero(as_tuple=False)[0].tolist()
        raise RuntimeError(f"[nan] get_rx phi @ {bad}")
    ####################
    if phi.dim() == 0:
        phi = phi.unsqueeze(0)
    c = torch.cos(phi / 2).unsqueeze(-1).unsqueeze(-1)
    s = torch.sin(phi / 2).unsqueeze(-1).unsqueeze(-1)
    if only_first_column:
        out = torch.cat([c, -1j * s], dim=1)
    else:
        row1 = torch.cat([c, -1j * s], dim=-1)
        row2 = torch.cat([-1j * s, c], dim=-1)
        out = torch.cat([row1, row2], dim=-2)
    ####### Protection #######
    if not torch.isfinite(out).all():
        bad = (~torch.isfinite(out)).nonzero(as_tuple=False)[0].tolist()
        raise RuntimeError(f"[nan] get_rx out @ {bad}")
    ####################
    return out


def get_ry(phi, only_first_column=False):
    """
    Compute Ry gate for each phi in the batch.
    Returns: tensor of shape [B, 2, 2]
    """
    # device = phi.device
    ####### Protection #######
    if not torch.isfinite(phi).all():
        bad = (~torch.isfinite(phi)).nonzero(as_tuple=False)[0].tolist()
        raise RuntimeError(f"[nan] get_ry phi @ {bad}")
    ####################
    if phi.dim() == 0:
        phi = phi.unsqueeze(0)
    # c = torch.cos(phi / 2).unsqueeze(-1).unsqueeze(-1).to(dtype=dev.dtype_complex)
    # s = torch.sin(phi / 2).unsqueeze(-1).unsqueeze(-1).to(dtype=dev.dtype_complex)
    c = torch.cos(phi / 2).unsqueeze(-1).unsqueeze(-1)
    s = torch.sin(phi / 2).unsqueeze(-1).unsqueeze(-1)
    dc = torch.complex128 if c.dtype == torch.float64 else torch.complex64
    c = c.to(dtype=dc)
    s = s.to(dtype=dc)
    if only_first_column:
        out = torch.cat([c, s], dim=1)
    else:
        row1 = torch.cat([c, -s], dim=-1)
        row2 = torch.cat([s, c], dim=-1)
        out = torch.cat([row1, row2], dim=-2)
    ####### Protection #######
    if not torch.isfinite(out).all():
        bad = (~torch.isfinite(out)).nonzero(as_tuple=False)[0].tolist()
        raise RuntimeError(f"[nan] get_ry out @ {bad}")
    ####################
    return out


def get_rot_gate(angles, only_first_column=False):
    """ Computes a composite rotation gate based on three angles per qubit.
    It uses the decomposition:
      R = Rz(angles[0]) · SX · Rz(angles[1]) · SX · Rz(angles[2])

    Args:
      angles (torch.Tensor): Either of shape [3] (unbatched) or [B, 3] (batched),
                             where B is the number of qubits.

    Returns:
      torch.Tensor: A 2x2 matrix if unbatched or [B, 2, 2] if batched.
    """
    # If the input is unbatched, add a batch dimension.
    unbatched = False
    if angles.dim() == 1:
        angles = angles.unsqueeze(0)  # now shape [1, 3]
        unbatched = True

    # Compute Rz for each angle component.
    # (Assuming your existing get_rz supports batched input:
    #  e.g. get_rz(angles[:, 0]) returns shape [B, 2, 2])
    rz1 = get_rz(angles[:, 0])
    rz2 = get_rz(angles[:, 1])
    ry2 = get_ry(angles[:, 1])
    rz3 = get_rz(angles[:, 2])

    # Compute the composite rotation using your multi_dim_matmul_reversed.
    # composite = multi_dim_matmul_reversed(rz1, SX_batch, rz2, SX_batch, rz3)
    composite = ops.multi_dim_matmul_reversed(rz1, ry2, rz3)  # using rz, ry, rz is the pennylane implementation

    if unbatched:
        composite = composite.squeeze(0)
    return composite
