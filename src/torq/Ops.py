import torch
import string
import math
from functools import lru_cache
from . import SingleQubitGates as single
from ._like import likeable


try:
    from cuquantum import contract
    _USE_CUQUANTUM = False
except ImportError:
    _USE_CUQUANTUM = False

_CNOT_PERM_CACHE = {}

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
    ####### Protection #######
    if not torch.isfinite(state).all(): raise RuntimeError("apply_matrix: state non-finite (in)")
    if not torch.isfinite(U).all():     raise RuntimeError("apply_matrix: U non-finite (in)")
    ####################
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
    ####### Protection #######
    if not torch.isfinite(out).all():   raise RuntimeError("apply_matrix: out non-finite")
    ####################
    return out

def apply_single_qubit_wall_batched(state, gates, n_qubits, qubit_order="msb"):
    """
    state: [B, 2**n] or [B, 2**n, 1] (complex/real)
    gates: [B, n_qubits, 2, 2]  (per-sample, per-qubit 2x2)
    returns: state with the wall applied, same shape as input
    """
    ####### Protection #######
    if not torch.isfinite(state).all(): raise RuntimeError("apply_single_qubit_wall_batched: state non-finite (in)")
    if not torch.isfinite(gates).all():     raise RuntimeError("apply_single_qubit_wall_batched: gates non-finite (in)")
    ####################
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

    # Reshape into an explicit tensor product layout: [B, 2, 2, ..., 2]
    psi = state.reshape(B, *([2] * n))
    perms = _qubit_permutations(n, qubit_order)

    for q in range(n):
        perm, inv = perms[q]
        if perm is None:
            psi_q = psi
        else:
            # bring the target qubit axis to position 1
            psi_q = psi.permute(perm)

        orig_shape = psi_q.shape
        psi_q = psi_q.reshape(B, 2, -1)      # [B, 2, M]

        if per_batch:
            g = gates[:, q]                  # [B, 2, 2]
            out = torch.bmm(g, psi_q)        # [B, 2, M]
        else:
            g = gates[q]                     # [2, 2]
            out = torch.matmul(g, psi_q)     # [B, 2, M] (broadcast)

        out = out.reshape(orig_shape)        # [B, 2, 2, ..., 2]

        if perm is None:
            psi = out
        else:
            psi = out.permute(inv)

    output = psi.reshape(B, dim)
    if squeeze_last:
        output = output.unsqueeze(-1)
    ####### Protection #######
    if not torch.isfinite(output).all():   raise RuntimeError("apply_single_qubit_wall_batched: output non-finite")
    ####################
    return output


@lru_cache(maxsize=None)
def _qubit_permutations(n_qubits: int, qubit_order: str):
    qubit_order = "msb" if qubit_order == "msb" else "lsb"
    perms = []
    for q in range(n_qubits):
        if qubit_order == "msb":
            axis = 1 + q
        else:  # lsb
            axis = 1 + (n_qubits - 1 - q)

        if axis == 1:
            perms.append((None, None))
            continue

        perm = (0, axis, *[i for i in range(1, n_qubits + 1) if i != axis])
        inv = [0] * (n_qubits + 1)
        for i, p in enumerate(perm):
            inv[p] = i
        perms.append((perm, tuple(inv)))
    return tuple(perms)


def apply_cnot_batched(state, n_qubits, control_qubit_id, target_qubit_id, qubit_order="msb"):
    """
    Apply a CNOT gate to a batched state via index permutation.
    state: [B, 2**n] or [B, 2**n, 1]
    """
    ####### Protection #######
    if not torch.isfinite(state).all(): raise RuntimeError("apply_cnot_batched: state non-finite (in)")
    ####################
    squeeze_last = False
    if state.dim() == 3 and state.shape[-1] == 1:
        squeeze_last = True
        state = state.squeeze(-1)

    B, dim = state.shape
    n = n_qubits
    if dim != (1 << n):
        raise ValueError("apply_cnot_batched: state length mismatch")

    perm = _get_cnot_perm(n, control_qubit_id, target_qubit_id, qubit_order, state.device)
    out = state.index_select(1, perm)

    if squeeze_last:
        out = out.unsqueeze(-1)
    ####### Protection #######
    if not torch.isfinite(out).all():   raise RuntimeError("apply_cnot_batched: output non-finite")
    ####################
    return out


def apply_cnot_ladder_batched(state, n_qubits, r=0, qubit_order="msb"):
    """
    Apply the same CNOT ladder as get_cnot_ladder, but directly on the state vector.
    """
    if n_qubits <= 1:
        return state
    for control_qubit_id in range(n_qubits):
        target_qubit_id = (control_qubit_id + r + 1) % n_qubits
        state = apply_cnot_batched(state, n_qubits, control_qubit_id, target_qubit_id, qubit_order=qubit_order)
    return state


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
