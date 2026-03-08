import math
import string
from typing import Any, Sequence

import torch

from .Dtypes import complex_dtype_like
from .SingleQubitGates import sigma_X_like, sigma_Y_like, sigma_Z_like
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

def _state_to_2d(state: torch.Tensor) -> torch.Tensor:
    if state.dim() == 3 and state.shape[-1] == 1:
        state = state.squeeze(-1)
    if state.dim() != 2:
        raise ValueError("state must have shape [B, 2**n] or [B, 2**n, 1].")
    return state


def _infer_n_qubits_from_state(state_2d: torch.Tensor) -> int:
    dim = state_2d.shape[1]
    n = int(math.log2(dim))
    if 2 ** n != dim:
        raise ValueError("State length must be a power of two.")
    return n


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
    state = _state_to_2d(state)
    B, _ = state.shape
    n = _infer_n_qubits_from_state(state)

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
        rho_q = torch.einsum("bki,bkj->bij", psi_q, psi_q.conj())  # [B,2,2]
        outs.append(torch.einsum("bij,ji->b", rho_q, obs[q]))
    out = torch.stack(outs, dim=1)
    return out.real if torch.is_complex(out) else out


def measure_local_Z(state: torch.Tensor) -> torch.Tensor:
    """
    Vectorized: compute ⟨Z⟩ on each qubit in one go.
    state: [B, 2**n_qubits]
    Returns: [B, n_qubits].
    """
    state = _state_to_2d(state)
    B, _ = state.shape
    n = _infer_n_qubits_from_state(state)

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


def _canonicalize_pauli_string(pauli: str) -> str:
    cleaned = pauli.upper().replace(" ", "")
    if not cleaned:
        raise ValueError("Pauli-string observable cannot be empty.")
    if cleaned.startswith("_") or cleaned.endswith("_") or "__" in cleaned:
        raise ValueError("Pauli-string observable cannot contain empty '_' groups.")
    bad = sorted(set(ch for ch in cleaned if ch not in {"I", "X", "Y", "Z", "_"}))
    if bad:
        raise ValueError(f"Unsupported Pauli characters {bad}. Supported characters: I, X, Y, Z, _.")
    return cleaned

def _split_pauli_groups(pauli: str) -> tuple[str, ...]:
    groups = tuple(pauli.split("_"))
    if any(not group for group in groups):
        raise ValueError("Pauli-string observable cannot contain empty '_' groups.")
    return groups


def _pauli_spec_is_all_identity(pauli: str) -> bool:
    chars = [ch for ch in pauli if ch != "_"]
    return bool(chars) and all(ch == "I" for ch in chars)


def _normalize_pauli_observable_string(observable: str, n_qubits: int) -> str:
    cleaned = _canonicalize_pauli_string(observable)
    if _pauli_spec_is_all_identity(cleaned):
        raise ValueError("Pauli-string observable cannot be all identity ('I').")

    for pauli_word in _split_pauli_groups(cleaned):
        if len(pauli_word) > n_qubits:
            raise ValueError(
                f"Pauli-string observables must have length <= n_qubits={n_qubits}. "
                f"Got length={len(pauli_word)}."
            )
    return cleaned


def _compile_pauli_word_on_qubits(
    pauli_word: str,
    qubits: Sequence[int],
    *,
    n_qubits: int,
    dtype: torch.dtype,
    device: torch.device,
    obs_idx: int,
) -> dict[str, Any]:
    if len(pauli_word) != len(qubits):
        raise ValueError(
            f"Observable #{obs_idx}: Pauli word length must match the number of qubits. "
            f"Got {len(pauli_word)} and {len(qubits)}."
        )

    flip_mask = 0
    yz_mask = 0
    n_y = 0
    for qubit, op in zip(qubits, pauli_word):
        if op not in {"I", "X", "Y", "Z"}:
            raise ValueError(
                f"Observable #{obs_idx}: unsupported Pauli operator {op!r}. "
                "Supported operators are I, X, Y, Z."
            )
        bit = 1 << (n_qubits - 1 - qubit)
        if op in {"X", "Y"}:
            flip_mask |= bit
        if op in {"Y", "Z"}:
            yz_mask |= bit
        if op == "Y":
            n_y += 1

    return {
        "kind": "pauli",
        "n_qubits": n_qubits,
        "qubits": tuple(qubits),
        "pauli_ops": tuple(pauli_word),
        "flip_mask": int(flip_mask),
        "yz_mask": int(yz_mask),
        "phase": torch.as_tensor(((-1j) ** n_y), dtype=dtype, device=device),
    }


def _compile_sliding_pauli_word(
    pauli_word: str,
    *,
    n_qubits: int,
    dtype: torch.dtype,
    device: torch.device,
    obs_idx_offset: int = 0,
) -> tuple[dict[str, Any], ...]:
    if len(pauli_word) > n_qubits:
        raise ValueError(
            f"Pauli-string observables must have length <= n_qubits={n_qubits}. "
            f"Got length={len(pauli_word)}."
        )

    compiled = []
    for start in range(n_qubits - len(pauli_word) + 1):
        compiled.append(
            _compile_pauli_word_on_qubits(
                pauli_word,
                qubits=tuple(range(start, start + len(pauli_word))),
                n_qubits=n_qubits,
                dtype=dtype,
                device=device,
                obs_idx=obs_idx_offset + len(compiled),
            )
        )
    return tuple(compiled)


def _is_matrix_like(value: Any) -> bool:
    if torch.is_tensor(value):
        return value.dim() >= 2
    try:
        t = torch.as_tensor(value)
    except Exception:
        return False
    return t.dim() >= 2


def _as_tensor_observable(value: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value).to(dtype=dtype, device=device)


def _measure_sliding_all_z_word(
    state_2d: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    n_qubits = _infer_n_qubits_from_state(state_2d)
    if window_size > n_qubits:
        raise ValueError(
            f"Pauli-string observables must have length <= n_qubits={n_qubits}. "
            f"Got length={window_size}."
        )
    if window_size == 1:
        return measure_local_Z(state_2d)

    probs = (state_2d.conj() * state_2d).real
    dim = probs.shape[1]
    basis = torch.arange(dim, device=state_2d.device, dtype=torch.long)
    bit_shifts = torch.arange(n_qubits - 1, -1, -1, device=state_2d.device, dtype=torch.long).unsqueeze(1)
    z_signs = 1 - 2 * ((basis.unsqueeze(0) >> bit_shifts) & 1).to(dtype=probs.dtype)
    window_signs = z_signs.unfold(0, window_size, 1).prod(dim=-1)
    return probs @ window_signs.transpose(0, 1)


def _measure_single_pauli_word(
    state_2d: torch.Tensor,
    pauli_word: str,
    *,
    pauli_chunk_size: int,
) -> torch.Tensor:
    n_qubits = _infer_n_qubits_from_state(state_2d)
    word = pauli_word.upper()

    if len(word) == 1:
        match word:
            case "Z":
                return measure_local_Z(state_2d)
            case "X":
                return measure_local_observable(
                    state_2d,
                    sigma_X_like(dtype=complex_dtype_like(state_2d), device=state_2d.device),
                )
            case "Y":
                return measure_local_observable(
                    state_2d,
                    sigma_Y_like(dtype=complex_dtype_like(state_2d), device=state_2d.device),
                )
            case "I":
                return state_2d.real.new_ones((state_2d.shape[0], n_qubits))

    if set(word) == {"Z"}:
        return _measure_sliding_all_z_word(state_2d, len(word))

    compiled = _compile_sliding_pauli_word(
        word,
        n_qubits=n_qubits,
        dtype=complex_dtype_like(state_2d),
        device=state_2d.device,
    )
    out = _measure_compiled_pauli_observables(
        state_2d=state_2d,
        pauli_specs=compiled,
        n_qubits=n_qubits,
        pauli_chunk_size=pauli_chunk_size,
    )
    return out.real if torch.is_complex(out) else out


def _measure_from_pauli_string(
    state_2d: torch.Tensor,
    observable: str,
    *,
    pauli_chunk_size: int,
) -> torch.Tensor:
    pauli = _normalize_pauli_observable_string(
        observable,
        _infer_n_qubits_from_state(state_2d),
    )

    outputs = [
        _measure_single_pauli_word(
            state_2d,
            pauli_word,
            pauli_chunk_size=pauli_chunk_size,
        )
        for pauli_word in _split_pauli_groups(pauli)
    ]
    return torch.cat(outputs, dim=1)


def _validate_hermitian_matrix(matrix: torch.Tensor, *, description: str) -> None:
    if not torch.allclose(matrix, matrix.conj().transpose(-2, -1), atol=1e-8, rtol=1e-5):
        raise ValueError(f"{description} must be Hermitian.")


def _measure_from_matrix(
    state_2d: torch.Tensor,
    observable: Any,
) -> torch.Tensor:
    n_qubits = _infer_n_qubits_from_state(state_2d)
    full_dim = 2 ** n_qubits
    state_complex = state_2d.to(dtype=complex_dtype_like(state_2d))
    obs = _as_tensor_observable(
        observable,
        dtype=state_complex.dtype,
        device=state_2d.device,
    )

    if obs.dim() == 2:
        if tuple(obs.shape) == (2, 2):
            _validate_hermitian_matrix(obs, description="Local observable")
            if _is_pauli_z_observable(obs, n_qubits, state_complex.dtype, state_2d.device):
                return measure_local_Z(state_2d)
            return measure_local_observable(state_2d, obs)
        if tuple(obs.shape) == (full_dim, full_dim):
            _validate_hermitian_matrix(obs, description="Global matrix observable")
            vals = torch.einsum("bi,ij,bj->b", state_complex.conj(), obs, state_complex)
            return vals.real.unsqueeze(1)
        raise ValueError(
            f"Matrix observable must have shape [2,2] or [{full_dim},{full_dim}] for n_qubits={n_qubits}. "
            f"Got {tuple(obs.shape)}."
        )

    if obs.dim() == 3:
        if tuple(obs.shape) == (n_qubits, 2, 2):
            _validate_hermitian_matrix(obs, description="Per-qubit local matrix observables")
            if _is_pauli_z_observable(obs, n_qubits, state_complex.dtype, state_2d.device):
                return measure_local_Z(state_2d)
            return measure_local_observable(state_2d, obs)
        if tuple(obs.shape[1:]) == (full_dim, full_dim):
            _validate_hermitian_matrix(obs, description="Matrix observables")
            vals = torch.einsum("bi,mij,bj->bm", state_complex.conj(), obs, state_complex)
            return vals.real
        raise ValueError(
            f"Matrix observables must have shape [n_qubits,2,2] or [m,{full_dim},{full_dim}] for n_qubits={n_qubits}. "
            f"Got {tuple(obs.shape)}."
        )

    raise ValueError(
        f"Matrix observables must be rank-2 or rank-3. Got rank={obs.dim()}."
    )


def _phase_signs_for_masks(
    basis_indices: torch.Tensor,
    yz_masks: torch.Tensor,
    n_qubits: int,
    real_dtype: torch.dtype,
) -> torch.Tensor:
    signs = torch.ones((yz_masks.shape[0], basis_indices.shape[0]), dtype=real_dtype, device=basis_indices.device)
    for bit_pos in range(n_qubits):
        active = ((yz_masks >> bit_pos) & 1).bool()
        if not torch.any(active):
            continue
        bit_values = ((basis_indices >> bit_pos) & 1).to(dtype=real_dtype)
        signs[active] = signs[active] * (1 - 2 * bit_values)
    return signs


def _measure_compiled_pauli_observables(
    state_2d: torch.Tensor,
    pauli_specs: Sequence[dict[str, Any]],
    n_qubits: int,
    pauli_chunk_size: int,
) -> torch.Tensor:
    state_complex = state_2d.to(dtype=complex_dtype_like(state_2d))
    B, dim = state_complex.shape
    basis = torch.arange(dim, device=state_complex.device, dtype=torch.long)
    bra = state_complex.conj().unsqueeze(1)  # [B, 1, dim]
    out_chunks = []

    for start in range(0, len(pauli_specs), pauli_chunk_size):
        chunk = pauli_specs[start:start + pauli_chunk_size]
        flip_masks = torch.tensor([spec["flip_mask"] for spec in chunk], device=state_complex.device, dtype=torch.long)
        yz_masks = torch.tensor([spec["yz_mask"] for spec in chunk], device=state_complex.device, dtype=torch.long)
        phases = torch.stack(
            [spec["phase"].to(dtype=state_complex.dtype, device=state_complex.device) for spec in chunk],
            dim=0,
        )  # [m]

        gather_idx = basis.unsqueeze(0) ^ flip_masks.unsqueeze(1)  # [m, dim]
        ket_transformed = state_complex[:, gather_idx]  # [B, m, dim]

        signs = _phase_signs_for_masks(
            basis_indices=basis,
            yz_masks=yz_masks,
            n_qubits=n_qubits,
            real_dtype=state_complex.real.dtype,
        )
        phase_per_basis = phases.unsqueeze(1) * signs.to(dtype=state_complex.dtype)  # [m, dim]
        out_chunks.append(torch.sum(bra * ket_transformed * phase_per_basis.unsqueeze(0), dim=-1))  # [B, m]

    return torch.cat(out_chunks, dim=1).reshape(B, -1)


def measure(
    state: torch.Tensor,
    observable: Any = None,
    *,
    pauli_chunk_size: int = 8,
) -> torch.Tensor:
    """
    Measure a state with one unified observable interface.

    Supported observables:
      - ``None``: default per-qubit Pauli-Z
      - Pauli string: one or more Pauli words separated by ``_``
      - Hermitian matrix or matrices:
        - ``[2,2]`` shared local observable
        - ``[n_qubits,2,2]`` per-qubit local observables
        - ``[2**n,2**n]`` one full-system observable
        - ``[m,2**n,2**n]`` multiple full-system observables
    """
    if pauli_chunk_size <= 0:
        raise ValueError("pauli_chunk_size must be >= 1.")

    state_2d = _state_to_2d(state)

    if observable is None:
        return measure_local_Z(state_2d)
    if isinstance(observable, str):
        return _measure_from_pauli_string(
            state_2d,
            observable,
            pauli_chunk_size=pauli_chunk_size,
        )
    if _is_matrix_like(observable):
        return _measure_from_matrix(state_2d, observable)
    raise TypeError(
        "observable must be None, a Pauli string, or a matrix/matrices tensor."
    )
