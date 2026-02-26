import math
import string
from collections import defaultdict
from typing import Any, Sequence

import torch

from .Dtypes import complex_dtype_like
from .SingleQubitGates import sigma_Z_like
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


def _normalize_wires(wires: Any, n_qubits: int, obs_idx: int) -> tuple[int, ...]:
    if isinstance(wires, int):
        wires = (wires,)
    elif isinstance(wires, Sequence):
        wires = tuple(wires)
    else:
        raise TypeError(f"Observable #{obs_idx}: 'wires' must be an int or a sequence of ints.")

    if not wires:
        raise ValueError(f"Observable #{obs_idx}: 'wires' cannot be empty.")

    normalized: list[int] = []
    for wire in wires:
        if not isinstance(wire, int):
            raise TypeError(f"Observable #{obs_idx}: all wires must be ints. Got {wire!r}.")
        if wire < 0 or wire >= n_qubits:
            raise ValueError(
                f"Observable #{obs_idx}: wire index {wire} is out of range for n_qubits={n_qubits}."
            )
        normalized.append(wire)

    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Observable #{obs_idx}: wires must be unique. Got {normalized}.")
    return tuple(normalized)


def _normalize_pauli_ops(pauli: Any, n_wires: int, obs_idx: int) -> tuple[str, ...]:
    if isinstance(pauli, str):
        cleaned = pauli.replace(" ", "").upper()
        if len(cleaned) == 1:
            ops = [cleaned] * n_wires
        elif len(cleaned) == n_wires:
            ops = list(cleaned)
        else:
            raise ValueError(
                f"Observable #{obs_idx}: pauli string length must be 1 or {n_wires}. "
                f"Got {len(cleaned)}."
            )
    elif isinstance(pauli, Sequence):
        ops = [str(op).upper() for op in pauli]
        if len(ops) != n_wires:
            raise ValueError(
                f"Observable #{obs_idx}: pauli sequence length must match wires length ({n_wires}). "
                f"Got {len(ops)}."
            )
        if any(len(op) != 1 for op in ops):
            raise ValueError(f"Observable #{obs_idx}: each pauli entry must be a single character in I/X/Y/Z.")
    else:
        raise TypeError(f"Observable #{obs_idx}: 'pauli' must be a string or a sequence of strings.")

    allowed = {"I", "X", "Y", "Z"}
    bad = [op for op in ops if op not in allowed]
    if bad:
        raise ValueError(
            f"Observable #{obs_idx}: unsupported Pauli ops {bad}. Supported ops are I, X, Y, Z."
        )
    return tuple(ops)


def _normalize_real_coeff(coeff: Any, dtype: torch.dtype, device: torch.device, obs_idx: int) -> torch.Tensor:
    coeff_t = torch.as_tensor(coeff, device=device)
    if coeff_t.numel() != 1:
        raise ValueError(f"Observable #{obs_idx}: 'coeff' must be a scalar.")
    coeff_t = coeff_t.reshape(())
    if torch.is_complex(coeff_t):
        imag = coeff_t.imag.abs().item()
        if imag > 1e-8:
            raise ValueError(f"Observable #{obs_idx}: 'coeff' must be real to keep the observable Hermitian.")
        coeff_t = coeff_t.real
    return coeff_t.to(dtype=dtype, device=device)


def _compile_measurement_observable(
    spec: dict[str, Any],
    n_qubits: int,
    dtype: torch.dtype,
    device: torch.device,
    obs_idx: int,
) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise TypeError(f"Observable #{obs_idx}: each observable must be a dict.")
    if "wires" not in spec:
        raise ValueError(f"Observable #{obs_idx}: missing required key 'wires'.")

    wires = _normalize_wires(spec["wires"], n_qubits=n_qubits, obs_idx=obs_idx)
    has_pauli = spec.get("pauli") is not None
    has_matrix = spec.get("matrix") is not None
    if has_pauli == has_matrix:
        raise ValueError(
            f"Observable #{obs_idx}: specify exactly one of 'pauli' or 'matrix'."
        )

    coeff = _normalize_real_coeff(spec.get("coeff", 1.0), dtype=dtype, device=device, obs_idx=obs_idx)

    if has_pauli:
        pauli_ops = _normalize_pauli_ops(spec["pauli"], n_wires=len(wires), obs_idx=obs_idx)
        flip_mask = 0
        yz_mask = 0
        n_y = 0
        for wire, op in zip(wires, pauli_ops):
            bit = 1 << (n_qubits - 1 - wire)
            if op in {"X", "Y"}:
                flip_mask |= bit
            if op in {"Y", "Z"}:
                yz_mask |= bit
            if op == "Y":
                n_y += 1
        # In this transformed-ket convention, each Y contributes a factor of -i.
        phase = coeff * ((-1j) ** n_y)
        return {
            "kind": "pauli",
            "n_qubits": n_qubits,
            "wires": wires,
            "flip_mask": int(flip_mask),
            "yz_mask": int(yz_mask),
            "phase": torch.as_tensor(phase, dtype=dtype, device=device),
        }

    matrix = torch.as_tensor(spec["matrix"]).to(dtype=dtype, device=device)
    local_dim = 1 << len(wires)
    expected_shape = (local_dim, local_dim)
    if tuple(matrix.shape) != expected_shape:
        raise ValueError(
            f"Observable #{obs_idx}: matrix must have shape {expected_shape} for wires={wires}. "
            f"Got {tuple(matrix.shape)}."
        )
    if not torch.allclose(matrix, matrix.conj().mT, atol=1e-8, rtol=1e-5):
        raise ValueError(f"Observable #{obs_idx}: matrix observable must be Hermitian.")
    return {
        "kind": "matrix",
        "n_qubits": n_qubits,
        "wires": wires,
        "matrix": matrix * coeff,
    }


@likeable
def compile_measurement_observables(
    observables: Sequence[dict[str, Any]],
    n_qubits: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(observables, Sequence) or isinstance(observables, (str, bytes)):
        raise TypeError("measurement observables must be provided as a sequence of dict specs.")
    if len(observables) == 0:
        raise ValueError("measurement observables cannot be empty.")
    return tuple(
        _compile_measurement_observable(
            spec=spec,
            n_qubits=n_qubits,
            dtype=dtype,
            device=device,
            obs_idx=obs_idx,
        )
        for obs_idx, spec in enumerate(observables)
    )


def _is_compiled_observable_specs(observables: Sequence[Any]) -> bool:
    if not observables:
        return False
    return all(
        isinstance(spec, dict) and {"kind", "n_qubits"} <= set(spec.keys())
        for spec in observables
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


def _reduced_density_matrix_on_wires(
    state_2d: torch.Tensor,
    wires: tuple[int, ...],
    n_qubits: int,
) -> torch.Tensor:
    state_complex = state_2d.to(dtype=complex_dtype_like(state_2d))
    B = state_complex.shape[0]
    k = len(wires)
    psi = state_complex.reshape(B, *([2] * n_qubits))
    moved_from = [1 + wire for wire in wires]
    moved_to = list(range(1 + n_qubits - k, 1 + n_qubits))
    psi_on_wires = psi.movedim(moved_from, moved_to).reshape(B, -1, 1 << k)
    return torch.einsum("bri,brj->bij", psi_on_wires, psi_on_wires.conj())  # [B, 2**k, 2**k]


def measure_observables(
    state: torch.Tensor,
    observables: Sequence[dict[str, Any]],
    *,
    pauli_chunk_size: int = 8,
) -> torch.Tensor:
    """
    Measure an arbitrary list of observables on one statevector simulation result.

    Each observable spec is a dict with:
      - required: ``wires`` (int or sequence[int])
      - exactly one of:
        - ``pauli``: Pauli operator(s) on wires (e.g. "Z", "ZZ", ["X", "Y"])
        - ``matrix``: Hermitian local matrix with shape [2**k, 2**k] for k=len(wires)
      - optional: ``coeff`` (real scalar)

    Returns: [B, n_observables]
    """
    if pauli_chunk_size <= 0:
        raise ValueError("pauli_chunk_size must be >= 1.")

    state_2d = _state_to_2d(state)
    n_qubits = _infer_n_qubits_from_state(state_2d)
    if not observables:
        raise ValueError("observables cannot be empty.")

    if _is_compiled_observable_specs(observables):
        compiled = tuple(observables)
    else:
        compiled = compile_measurement_observables(
            observables=observables,
            n_qubits=n_qubits,
            x=state_2d,
        )

    B = state_2d.shape[0]
    out = state_2d.new_zeros((B, len(compiled)), dtype=complex_dtype_like(state_2d))

    pauli_cols: list[int] = []
    pauli_specs: list[dict[str, Any]] = []
    matrix_groups: dict[tuple[int, ...], list[tuple[int, torch.Tensor]]] = defaultdict(list)

    for col, spec in enumerate(compiled):
        if spec.get("n_qubits") != n_qubits:
            raise ValueError(
                f"Observable #{col} was compiled for n_qubits={spec.get('n_qubits')} "
                f"but state has n_qubits={n_qubits}."
            )
        kind = spec.get("kind")
        if kind == "pauli":
            pauli_cols.append(col)
            pauli_specs.append(spec)
        elif kind == "matrix":
            matrix_groups[tuple(spec["wires"])].append((col, spec["matrix"]))
        else:
            raise ValueError(f"Observable #{col} has unsupported compiled kind: {kind!r}.")

    if pauli_specs:
        pauli_vals = _measure_compiled_pauli_observables(
            state_2d=state_2d,
            pauli_specs=pauli_specs,
            n_qubits=n_qubits,
            pauli_chunk_size=pauli_chunk_size,
        )
        out[:, pauli_cols] = pauli_vals

    if matrix_groups:
        state_complex_dtype = complex_dtype_like(state_2d)
        for wires, entries in matrix_groups.items():
            rho = _reduced_density_matrix_on_wires(state_2d=state_2d, wires=wires, n_qubits=n_qubits)
            mats = torch.stack(
                [mat.to(dtype=state_complex_dtype, device=state_2d.device) for _, mat in entries],
                dim=0,
            )  # [m, 2**k, 2**k]
            vals = torch.einsum("bij,mji->bm", rho, mats)
            cols = [col for col, _ in entries]
            out[:, cols] = vals

    return out.real if torch.is_complex(out) else out


def measure(state: torch.Tensor, observable: torch.Tensor | None = None) -> torch.Tensor:
    """
    Squeeze state’s trailing singleton (if present) and compute per-qubit local expectations.
    """
    state_2d = _state_to_2d(state)
    n = _infer_n_qubits_from_state(state_2d)

    if observable is None or _is_pauli_z_observable(
        observable,
        n_qubits=n,
        dtype=complex_dtype_like(state_2d),
        device=state_2d.device,
    ):
        return measure_local_Z(state_2d)
    return measure_local_observable(state_2d, observable)
