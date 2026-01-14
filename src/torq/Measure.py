import torch
import math
import string


def measure_local(state: torch.Tensor, local_observable: torch.Tensor) -> torch.Tensor:
    """
    Vectorized: compute ⟨Z⟩ on each qubit in one go.
    state: [B, 2**n_qubits]
    local_observable must be diag(+1,-1) (sigma_Z).
    Returns: [B, n_qubits].
    """
    B, dim = state.shape
    n = int(math.log2(dim))

    # 1) probabilities over each basis amplitude
    probs = (state.conj() * state).view(B, *([2] * n)).real  # [B,2,2,...,2]

    # 2) build Z_axes: shape [n,2,2,...,2]
    pauli_z = torch.tensor([1.0, -1.0], device=probs.device, dtype=probs.dtype)
    full_shape = [2] * n
    Z_axes = []
    for q in range(n):
        vs = [1] * n
        vs[q] = 2
        Zq = pauli_z.view(vs).expand(*full_shape)
        Z_axes.append(Zq)
    Z_axes = torch.stack(Z_axes, dim=0)  # [n,2,2,...,2]

    # 3) pick n distinct letters that aren’t 'b' or 'q'
    letters = [c for c in string.ascii_lowercase if c not in ('b','q')]
    idx = ''.join(letters[:n])           # e.g. 'acdef' for 5 qubits

    # equation: 'bacdef...,qacdef...->bq'
    eq = f"b{idx},q{idx}->bq"
    return torch.einsum(eq, probs, Z_axes)


def measure_local_Z(state: torch.Tensor) -> torch.Tensor:
    # state: [B, 2**n]
    B, dim = state.shape
    n = int(math.log2(dim))
    assert 2**n == dim, "State length must be a power of two"
    probs = (state.conj() * state).real.view(B, *([2]*n))  # [B,2,...,2]
    outs = []
    for q in range(n):
        # move qubit-q axis to the end, flatten the rest
        p = probs.movedim(1+q, -1).reshape(B, -1, 2)       # [B, dim/2, 2]
        outs.append((p[..., 0] - p[..., 1]).sum(dim=1))    # [B]
    return torch.stack(outs, dim=1)                        # [B, n]


def measure(state: torch.Tensor, observable: torch.Tensor) -> torch.Tensor:
    """
    Squeeze state’s trailing singleton (if present) and vectorized measure.
    """
    if state.dim() == 3 and state.shape[2] == 1:
        state = state.squeeze(-1)
    return measure_local(state, observable)
