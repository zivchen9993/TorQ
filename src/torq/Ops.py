import math
import os
import string

import torch

from . import SingleQubitGates as single
from ._like import likeable


try:
    from cuquantum import contract
    _USE_CUQUANTUM = False
except ImportError:
    _USE_CUQUANTUM = False

_CNOT_PERM_CACHE = {}
_QUBIT_PAIR_INDEX_CACHE = {}
_RUNTIME_CHECKS = os.getenv("TORQ_RUNTIME_CHECKS", "").strip().lower() in {"1", "true", "yes", "on"}


def runtime_checks_enabled() -> bool:
    return _RUNTIME_CHECKS


def _raise_if_nonfinite(name: str, tensor: torch.Tensor) -> None:
    if runtime_checks_enabled() and not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name}: encountered non-finite values")

def multi_dim_tensor_product(*args):
    """
    Compute the Kronecker product of multiple matrices in a batched manner,
    using tensor slicing with broadcasting (via einsum) to avoid iterative Python loops.

    Each input can be unbatched (2D) or batched (3D).
    The result is a tensor of shape [B, prod(rows), prod(cols)].
    """
    # Ensure every matrix is batched.
    res_batched = True
    if args[0].dim() == 2:
        res_batched = False
    mats = [x if x.dim() == 3 else x.unsqueeze(0) for x in args]
    B = mats[0].shape[0]
    k = len(mats)

    # Get the row and column dimensions for each matrix.
    row_dims = [A.shape[1] for A in mats]
    col_dims = [A.shape[2] for A in mats]

    # We need unique subscripts for each matrix's row and column indices.
    # For simplicity we use the first 2*k letters of the alphabet.
    letters = string.ascii_lowercase.replace("b", "")
    if 2 * k > len(letters):
        raise ValueError("Too many matrices for available einsum subscripts.")
    row_indices = letters[:k]
    col_indices = letters[k:2 * k]

    # Build the einsum string:
    # - For each matrix, we associate a subscript for the row and column.
    #   For example, for k=2, we want something like: "bij, bkl -> b ik jl" which then will
    #   be reshaped to [B, (i*k), (j*l)].
    #
    # Here we join the subscripts for each matrix: e.g., "b{i}{j}".
    input_subscripts = [f"b{r}{c}" for r, c in zip(row_indices, col_indices)]
    output_subscript = "b" + "".join(row_indices) + "".join(col_indices)
    einsum_str = ",".join(input_subscripts) + "->" + output_subscript

    # Compute the einsum.
    result = torch.einsum(einsum_str, *mats)

    # Reshape to merge all row and all column indices.
    out_rows = math.prod(row_dims)
    out_cols = math.prod(col_dims)
    result = result.reshape(B, int(out_rows), int(out_cols))
    if not res_batched:
        result = result.squeeze(0)
    return result


def multi_dim_matmul_reversed(*args):
    """
    Like multi_dim_matmul, but multiplies in reverse order,
    and if all inputs were 2D returns a 2D result.
    """
    # Move everything to the last argument's device
    target_device = args[-1].device
    mats = [x.to(target_device) if isinstance(x, torch.Tensor) else x
            for x in args]

    # If every argument is 2D, we'll want to squeeze the batch dim at the end
    unbatched = all(mat.dim() == 2 for mat in mats)

    # Ensure each tensor has a batch dimension
    mats = [mat if mat.dim() == 3 else mat.unsqueeze(0) for mat in mats]

    # Reverse the order and do a straight left‑to‑right batched matmul
    mats = list(reversed(mats))
    result = mats[0]
    for M in mats[1:]:
        result = torch.matmul(result, M)

    # If originally unbatched, drop the leading batch dim
    if unbatched:
        result = result.squeeze(0)

    return result


@likeable
def kron_with_replace(n_qubits, replacements, *, dtype, device):
    """
    Constructs the Kronecker (tensor) product for an n_qubit system, using identities (I)
     everywhere except at positions specified by the dictionary 'replacements'.
    The keys in 'replacements' should be integer indices (0-indexed) and
    the corresponding value is the 2×2 operator to put at that qubit. """
    I = single.ID1_like(dtype=dtype, device=device)
    # Build the list of matrices to tensor together:
    mats = [replacements[i] if i in replacements else I for i in range(n_qubits)]
    return multi_dim_tensor_product(*mats)


def apply_matrix(state: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Apply a full [2**n × 2**n] unitary U to batched state [B,2**n],
    using cuQuantum.contract if available, otherwise torch.matmul.
    """
    device = U.device
    _raise_if_nonfinite("apply_matrix(state)", state)
    _raise_if_nonfinite("apply_matrix(U)", U)
    # If U got returned as [1,dim,dim], drop that leading 1
    if U.dim() == 3 and U.shape[0] == 1:
           U = U.squeeze(0)  # now [dim,dim]
     # Now U is [dim,dim], state is [B,dim]
    if _USE_CUQUANTUM:
        # single cuTensorNet contraction instead of many tiny kernels
        out = contract('bj,ji->bi', state, U).to(device)
    else:
        # fallback to a single big matmul
        out = torch.matmul(state, U.T).to(device)
    _raise_if_nonfinite("apply_matrix(out)", out)
    return out


def _get_qubit_pair_indices(n_qubits: int, qubit_id: int, qubit_order: str, device: torch.device):
    order = "msb" if qubit_order == "msb" else "lsb"
    key = (n_qubits, qubit_id, order, device.type, device.index)
    cached = _QUBIT_PAIR_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    dim = 1 << n_qubits
    idx = torch.arange(dim, device=device)
    if order == "msb":
        stride = 1 << (n_qubits - 1 - qubit_id)
    else:
        stride = 1 << qubit_id
    i0 = idx[(idx // stride) % 2 == 0]
    i1 = i0 + stride
    _QUBIT_PAIR_INDEX_CACHE[key] = (i0, i1)
    return i0, i1


def apply_single_qubit_wall_batched(state, gates, n_qubits, qubit_order="msb"):
    """
    state: [B, 2**n] or [B, 2**n, 1] (complex/real)
    gates: [B, n_qubits, 2, 2]  (per-sample, per-qubit 2x2)
    returns: state with the wall applied, same shape as input
    """
    _raise_if_nonfinite("apply_single_qubit_wall_batched(state)", state)
    _raise_if_nonfinite("apply_single_qubit_wall_batched(gates)", gates)
    squeeze_last = False
    if state.dim() == 3 and state.shape[-1] == 1:
        squeeze_last = True
        state = state.squeeze(-1)               # [B, 2**n]

    B, dim = state.shape
    n = n_qubits
    if dim != (1 << n):
        raise ValueError("apply_single_qubit_wall_batched: state length mismatch")

    # gates: [B, n, 2, 2] or [n, 2, 2]
    if gates.dim() == 4:
        per_batch = True
        if gates.shape[0] != B or gates.shape[1] != n:
            raise ValueError("apply_single_qubit_wall_batched: gates shape mismatch")
    elif gates.dim() == 3:
        per_batch = False
        if gates.shape[0] != n:
            raise ValueError("apply_single_qubit_wall_batched: gates shape mismatch")
    else:
        raise ValueError("apply_single_qubit_wall_batched: gates must be [B,n,2,2] or [n,2,2]")

    for q in range(n):
        i0, i1 = _get_qubit_pair_indices(n, q, qubit_order, state.device)

        v = torch.stack([state[:, i0], state[:, i1]], dim=2)      # [B, M, 2]
        v = v.transpose(1, 2)                                     # [B, 2, M]
        if per_batch:
            g = gates[:, q]                                       # [B, 2, 2]
            out = torch.matmul(g, v)                              # [B, 2, M]
        else:
            g = gates[q]                                          # [2, 2]
            out = torch.matmul(g, v)                              # [B, 2, M]
        out = out.transpose(1, 2).contiguous()                    # [B, M, 2]

        # scatter back
        state[:, i0] = out[:, :, 0]
        state[:, i1] = out[:, :, 1]

    output = state
    if squeeze_last:
        output = output.unsqueeze(-1)
    _raise_if_nonfinite("apply_single_qubit_wall_batched(out)", output)
    return output


def apply_cnot(state: torch.Tensor, n_qubits: int, control_qubit_id: int, target_qubit_id: int, qubit_order="msb") -> torch.Tensor:
    squeeze_last = False
    if state.dim() == 3 and state.shape[-1] == 1:
        squeeze_last = True
        state = state.squeeze(-1)
    if state.dim() != 2 or state.shape[1] != (1 << n_qubits):
        raise ValueError("apply_cnot: state must have shape [B, 2**n] or [B, 2**n, 1].")

    perm = _get_cnot_perm(
        n_qubits=n_qubits,
        control_qubit_id=control_qubit_id,
        target_qubit_id=target_qubit_id,
        qubit_order=qubit_order,
        device=state.device,
    )
    out = state[:, perm].contiguous()
    if squeeze_last:
        out = out.unsqueeze(-1)
    return out


def apply_cnot_ladder(state: torch.Tensor, n_qubits: int, r: int = 0, qubit_order="msb") -> torch.Tensor:
    if (r + 1) == n_qubits:
        return state
    out = state
    for control_qubit_id in range(n_qubits):
        target_qubit_id = (control_qubit_id + r + 1) % n_qubits
        out = apply_cnot(
            out,
            n_qubits=n_qubits,
            control_qubit_id=control_qubit_id,
            target_qubit_id=target_qubit_id,
            qubit_order=qubit_order,
        )
    return out


def _get_cnot_perm(n_qubits, control_qubit_id, target_qubit_id, qubit_order, device):
    qubit_order = "msb" if qubit_order == "msb" else "lsb"
    key = (n_qubits, control_qubit_id, target_qubit_id, qubit_order, device.type, device.index)
    perm = _CNOT_PERM_CACHE.get(key)
    if perm is not None:
        return perm

    idx = torch.arange(1 << n_qubits, dtype=torch.long)
    if qubit_order == "msb":
        ctrl_bit = 1 << (n_qubits - 1 - control_qubit_id)
        tgt_bit = 1 << (n_qubits - 1 - target_qubit_id)
    else:
        ctrl_bit = 1 << control_qubit_id
        tgt_bit = 1 << target_qubit_id

    mask = (idx & ctrl_bit) != 0
    perm = idx.clone()
    perm[mask] ^= tgt_bit
    perm = perm.to(device=device)

    _CNOT_PERM_CACHE[key] = perm
    return perm
