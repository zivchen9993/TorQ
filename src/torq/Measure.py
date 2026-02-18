import torch
import math
import string
from .SingleQubitGates import sigma_Z_like
from .Dtypes import complex_dtype_like
from ._like import likeable


@likeable
def local_obs_like(obs, *, dtype, device):
    if obs is None:
        raise ValueError("local observables cannot be None")
    obs = torch.as_tensor(obs)
    if obs.shape != (2, 2):
        raise ValueError("local observables are 2x2 tensors")
    # Hermitian check: O == O^\dagger (conjugate transpose).
    if not torch.allclose(obs, obs.conj().mT, atol=1e-8, rtol=1e-5):
        raise ValueError("local observables must be Hermitian, but the provided observable is not conjugate-transpose of itself")
    return obs.to(dtype=dtype, device=device)


def _is_pauli_z_observable(observable: torch.Tensor, n_qubits: int, dtype: torch.dtype, device: torch.device) -> bool:
    obs = observable.to(dtype=dtype, device=device)
    pauli_z = sigma_Z_like(dtype=dtype, device=device)
    if obs.dim() == 2:
        return tuple(obs.shape) == (2, 2) and torch.allclose(obs, pauli_z, atol=1e-8, rtol=1e-5)
    if obs.dim() == 3:
        target = pauli_z.unsqueeze(0).expand(n_qubits, -1, -1)
        return tuple(obs.shape) == (n_qubits, 2, 2) and torch.allclose(obs, target, atol=1e-8, rtol=1e-5)
    return False


def _normalize_local_observable(local_observable: torch.Tensor | None, n_qubits: int,
                                dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if local_observable is None:
        return sigma_Z_like(dtype=dtype, device=device).unsqueeze(0).expand(n_qubits, -1, -1)

    obs = local_observable.to(dtype=dtype, device=device)
    if obs.dim() == 2:
        if tuple(obs.shape) != (2, 2):
            raise ValueError("A shared local observable must have shape [2, 2].")
        obs = obs.unsqueeze(0).expand(n_qubits, -1, -1)
    elif obs.dim() == 3:
        if tuple(obs.shape) != (n_qubits, 2, 2):
            raise ValueError(
                f"A per-qubit local observable must have shape [{n_qubits}, 2, 2]. "
                f"Got: {tuple(obs.shape)}."
            )
    else:
        raise ValueError(
            "local_observable must be None, [2,2], or [n_qubits,2,2]."
        )
    return obs


def measure_local_observable(state: torch.Tensor, local_observable: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute per-qubit local expectation values.

    state: [B, 2**n_qubits] (or [B, 2**n_qubits, 1], trailing singleton allowed)
    local_observable:
      - None: defaults to Pauli-Z on all qubits
      - [2,2]: same observable on all qubits
      - [n_qubits,2,2]: per-qubit observables

    Returns: [B, n_qubits]
    """
    if state.dim() == 3 and state.shape[-1] == 1:
        state = state.squeeze(-1)
    if state.dim() != 2:
        raise ValueError("state must have shape [B, 2**n] or [B, 2**n, 1].")

    B, dim = state.shape
    n = int(math.log2(dim))
    if 2 ** n != dim:
        raise ValueError("State length must be a power of two.")

    obs = _normalize_local_observable(
        local_observable,
        n_qubits=n,
        dtype=complex_dtype_like(state),
        device=state.device,
    )

    psi = state.reshape(B, *([2] * n))
    outs = []
    for q in range(n):
        psi_q = psi.movedim(1 + q, -1).reshape(B, -1, 2)  # [B, 2**(n-1), 2]
        rho_q = torch.einsum("bki,bkj->bij", psi_q.conj(), psi_q)  # [B,2,2]
        outs.append(torch.einsum("bij,ji->b", rho_q, obs[q]))
    out = torch.stack(outs, dim=1)
    return out.real if torch.is_complex(out) else out


def measure_local_Z(state: torch.Tensor) -> torch.Tensor:
    """
    Vectorized: compute ⟨Z⟩ on each qubit in one go.
    state: [B, 2**n_qubits]
    Returns: [B, n_qubits].
    """
    if state.dim() == 3 and state.shape[-1] == 1:
        state = state.squeeze(-1)
    if state.dim() != 2:
        raise ValueError("state must have shape [B, 2**n] or [B, 2**n, 1].")

    B, dim = state.shape
    n = int(math.log2(dim))
    if 2 ** n != dim:
        raise ValueError("State length must be a power of two.")

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
    letters = [c for c in string.ascii_lowercase if c not in ('b', 'q')]
    idx = ''.join(letters[:n])  # e.g. 'acdef' for 5 qubits

    # equation: 'bacdef...,qacdef...->bq'
    eq = f"b{idx},q{idx}->bq"
    return torch.einsum(eq, probs, Z_axes)


def measure(state: torch.Tensor, observable: torch.Tensor | None = None) -> torch.Tensor:
    """
    Squeeze state’s trailing singleton (if present) and compute per-qubit local expectations.
    """
    if state.dim() == 3 and state.shape[-1] == 1:
        state_2d = state.squeeze(-1)
    else:
        state_2d = state
    if state_2d.dim() != 2:
        raise ValueError("state must have shape [B, 2**n] or [B, 2**n, 1].")
    n = int(math.log2(state_2d.shape[1]))
    if 2 ** n != state_2d.shape[1]:
        raise ValueError("State length must be a power of two.")

    if observable is None or _is_pauli_z_observable(
        observable,
        n_qubits=n,
        dtype=complex_dtype_like(state_2d),
        device=state_2d.device,
    ):
        return measure_local_Z(state_2d)
    return measure_local_observable(state_2d, observable)
