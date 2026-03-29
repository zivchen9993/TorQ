from __future__ import annotations

import importlib
import inspect
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.util import find_spec

import torch

from .Measure import _is_pauli_z_observable
from .Templates import get_angle_embedding_sigmas


@dataclass
class _PennyLaneRunner:
    circuit: callable | None
    angle_scaling_method: str
    angle_scaling: float
    basis_angle_embedding: str
    n_outputs: int
    measure_state: callable | None = None
    measurement_circuit: callable | None = None
    measurement_result_formatter: callable | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, scaled, _, _ = get_angle_embedding_sigmas(
            x,
            angle_scaling_method=self.angle_scaling_method,
            angle_scaling=self.angle_scaling,
            basis=self.basis_angle_embedding,
        )
        if self.measurement_circuit is not None:
            measured = self.measurement_circuit(scaled)
            return _format_pennylane_measurement_result(
                measured,
                formatter=self.measurement_result_formatter,
            ).to(torch.float32)

        if self.circuit is None:
            raise RuntimeError("PennyLane runner requires either a measurement_circuit or circuit.")
        out = self.circuit(scaled)
        measured_state = _maybe_measure_pennylane_state(
            out,
            measure_state=self.measure_state,
        )
        if measured_state is not None:
            return measured_state.to(torch.float32)
        return _normalize_pennylane_output(
            out,
            batch_size=scaled.shape[0],
            n_outputs=self.n_outputs,
        ).to(torch.float32)


def maybe_create_pennylane_backend(layer):
    if not getattr(layer.config, "pennylane_backend", False):
        return None

    if not _pennylane_dependencies_available():
        warnings.warn(
            "pennylane_backend=True but torq_bench and/or pennylane is not installed; "
            "using TorQ backend.",
            RuntimeWarning,
        )
        return None

    comparison_cls = _load_pennylane_comparison_class()
    if comparison_cls is None:
        return None

    if not _backend_supports_requested_basis(comparison_cls, layer.basis_angle_embedding):
        warnings.warn(
            "pennylane_backend=True but the selected PennyLane backend does not support "
            f"basis_angle_embedding={layer.basis_angle_embedding!r}; using TorQ backend.",
            RuntimeWarning,
        )
        return None

    observable = getattr(layer, "observables", None)
    if not _backend_supports_requested_observables(
        comparison_cls,
        observable,
        n_qubits=layer.n_qubits,
    ):
        warnings.warn(
            "pennylane_backend=True but the selected PennyLane backend does not support "
            "the requested observables; using TorQ backend.",
            RuntimeWarning,
        )
        return None

    dev_name = getattr(layer.config, "pennylane_dev_name", "default.qubit") or "default.qubit"
    backend = _create_pennylane_comparison_backend(
        comparison_cls,
        layer=layer,
        dev_name=dev_name,
    )

    measurement_circuit = _select_pennylane_measurement_circuit(backend, layer.ansatz_name)
    circuit = None if measurement_circuit is not None else _select_pennylane_circuit(
        backend,
        layer.ansatz_name,
        layer.data_reupload_every,
    )

    if measurement_circuit is None and circuit is None:
        warnings.warn(
            f"pennylane_backend=True but ansatz_name='{layer.ansatz_name}' is not supported; "
            "using TorQ backend.",
            RuntimeWarning,
        )
        return None

    return _PennyLaneRunner(
        circuit=circuit,
        angle_scaling_method=layer.angle_scaling_method,
        angle_scaling=layer.angle_scaling,
        basis_angle_embedding=layer.basis_angle_embedding,
        n_outputs=layer.n_qubits,
        measure_state=getattr(backend, "measure_state", None),
        measurement_circuit=measurement_circuit,
        measurement_result_formatter=getattr(backend, "_format_batched_measurement_result", None),
    )


def _pennylane_dependencies_available() -> bool:
    return find_spec("torq_bench") is not None and find_spec("pennylane") is not None


def _load_pennylane_comparison_class():
    try:
        module = importlib.import_module("torq_bench.PennyLaneComparison")
    except Exception as exc:
        warnings.warn(
            f"Failed to import torq_bench.PennyLaneComparison ({exc}); using TorQ backend.",
            RuntimeWarning,
        )
        return None
    if hasattr(module, "PennyLaneComparison"):
        return module.PennyLaneComparison
    if hasattr(module, "qml_sanity_check"):
        return module.qml_sanity_check
    warnings.warn(
        "torq_bench.PennyLaneComparison does not expose PennyLaneComparison or qml_sanity_check; "
        "using TorQ backend.",
        RuntimeWarning,
    )
    return None


def _comparison_accepts_parameter(comparison_cls, parameter_name: str) -> bool:
    try:
        parameters = inspect.signature(comparison_cls).parameters.values()
    except (TypeError, ValueError):
        return False

    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == parameter_name
        for parameter in parameters
    )


def _backend_supports_requested_basis(comparison_cls, basis_angle_embedding: str) -> bool:
    return (
        (basis_angle_embedding or "X").upper() in ("X", "RX")
        or _comparison_accepts_parameter(comparison_cls, "basis_angle_embedding")
    )


def _supports_default_pauli_z_observable(
    observable: torch.Tensor | str | object | None,
    *,
    n_qubits: int,
) -> bool:
    if observable is None:
        return True
    if isinstance(observable, str):
        return observable.replace(" ", "").upper() == "Z"

    try:
        tensor = observable if torch.is_tensor(observable) else torch.as_tensor(observable)
    except Exception:
        return False

    return _is_pauli_z_observable(
        tensor,
        n_qubits=n_qubits,
        dtype=tensor.to(dtype=torch.complex64).dtype,
        device=tensor.device,
    )


def _backend_supports_requested_observables(
    comparison_cls,
    observable: torch.Tensor | str | object | None,
    *,
    n_qubits: int,
) -> bool:
    return _supports_default_pauli_z_observable(
        observable,
        n_qubits=n_qubits,
    ) or _comparison_accepts_parameter(comparison_cls, "observables")


def _create_pennylane_comparison_backend(comparison_cls, *, layer, dev_name: str):
    kwargs = {
        "n_qubits": layer.n_qubits,
        "n_layers": layer.n_layers,
        "weights": layer.params,
        "weights_last_layer_data_re": getattr(layer, "params_last_layer_reupload", None),
        "data_reupload_every": layer.data_reupload_every,
        "pennylane_dev_name": dev_name,
    }
    optional_kwargs = {
        "basis_angle_embedding": layer.basis_angle_embedding,
        "observables": getattr(layer, "observables", None),
        "pauli_measurement_chunk_size": getattr(layer.config, "pauli_measurement_chunk_size", 8),
        "config": layer.config,
    }
    for name, value in optional_kwargs.items():
        if _comparison_accepts_parameter(comparison_cls, name):
            kwargs[name] = value
    return comparison_cls(**kwargs)


def _select_pennylane_measurement_circuit(pennylane_backend, ansatz_name: str):
    if not hasattr(pennylane_backend, "build_measurement_circuit"):
        return None
    try:
        return pennylane_backend.build_measurement_circuit(ansatz_name)
    except ValueError as exc:
        if "does not support ansatz_name" in str(exc):
            return None
        raise


def _select_pennylane_circuit(pennylane_backend, ansatz_name: str, data_reupload_every: int):
    if data_reupload_every:
        candidates = {
            "basic_entangling": ("data_re_circuit_basic_entangling",),
            "single_rot_basic_ent": ("data_re_circuit_single_rot_basic_ent",),
            "strongly_entangling": ("data_re_circuit_strongly_entangling",),
            "cross_mesh": ("data_re_circuit_cross_mesh",),
            "cross_mesh_2_rots": ("data_re_circuit_cross_mesh_2_rots",),
            "cross_mesh_cx_rot": ("data_re_circuit_cross_mesh_cx_rot",),
            "tile": ("data_re_circuit_tile",),
            "no_entanglement_ansatz": (
                "data_re_circuit_no_entanglement_ansatz",
                "data_re_circuit_no_entanglement",
            ),
            "no_entanglement": (
                "data_re_circuit_no_entanglement_ansatz",
                "data_re_circuit_no_entanglement",
            ),
        }
    else:
        candidates = {
            "basic_entangling": ("circuit_basic_entangling",),
            "single_rot_basic_ent": ("circuit_single_rot_basic_ent",),
            "strongly_entangling": ("circuit_strongly_entangling",),
            "cross_mesh": ("circuit_cross_mesh",),
            "cross_mesh_2_rots": ("circuit_cross_mesh_2_rots",),
            "cross_mesh_cx_rot": ("circuit_cross_mesh_cx_rot",),
            "tile": ("circuit_tile",),
            "no_entanglement_ansatz": (
                "circuit_no_entanglement_ansatz",
                "circuit_no_entanglement",
            ),
            "no_entanglement": (
                "circuit_no_entanglement_ansatz",
                "circuit_no_entanglement",
            ),
        }

    for method_name in candidates.get(ansatz_name, ()):
        if hasattr(pennylane_backend, method_name):
            return getattr(pennylane_backend, method_name)()
    return None


def _format_pennylane_measurement_result(
    result: torch.Tensor | Sequence[torch.Tensor],
    *,
    formatter: callable | None,
) -> torch.Tensor:
    if formatter is not None:
        formatted = formatter(result)
        if isinstance(formatted, torch.Tensor):
            return formatted
        return torch.as_tensor(formatted)

    if isinstance(result, torch.Tensor):
        tensor = torch.real(result)
        if tensor.dim() == 0:
            return tensor.reshape(1, 1)
        if tensor.dim() == 1:
            return tensor.unsqueeze(-1)
        return tensor

    return torch.real(torch.stack([torch.as_tensor(value) for value in result], dim=-1))


def _normalize_pennylane_output(
    output: torch.Tensor | Sequence[torch.Tensor],
    *,
    batch_size: int,
    n_outputs: int,
) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        tensor = output
    else:
        output = tuple(output)
        if not output:
            return torch.empty(batch_size, 0)
        tensor = torch.stack(output, dim=0)

    if tensor.dim() == 0:
        if batch_size == 1 and n_outputs == 1:
            return tensor.reshape(1, 1)
        raise ValueError(
            "PennyLane circuit returned a scalar, but the backend expected "
            f"{n_outputs} outputs for batch size {batch_size}."
        )

    if tensor.dim() == 1:
        if batch_size == 1 and tensor.shape[0] == n_outputs:
            return tensor.unsqueeze(0)
        if n_outputs == 1 and tensor.shape[0] == batch_size:
            return tensor.unsqueeze(1)
        raise ValueError(
            "PennyLane circuit returned a 1D tensor with shape "
            f"{tuple(tensor.shape)}, which cannot be normalized to "
            f"({batch_size}, {n_outputs})."
        )

    if tensor.dim() != 2:
        raise ValueError(
            "PennyLane circuit returned a tensor with shape "
            f"{tuple(tensor.shape)}, but only scalar, vector, and matrix outputs "
            "are supported."
        )

    # PennyLane stacks same-shaped measurements along the leading dimension, so
    # a batched expectation vector arrives as [n_outputs, batch_size].
    if tensor.shape == (n_outputs, batch_size):
        return tensor.transpose(0, 1)
    if tensor.shape == (batch_size, n_outputs):
        return tensor

    raise ValueError(
        "PennyLane circuit returned a tensor with shape "
        f"{tuple(tensor.shape)}, which cannot be normalized to "
        f"({batch_size}, {n_outputs})."
    )


def _maybe_measure_pennylane_state(
    output: torch.Tensor | Sequence[torch.Tensor],
    *,
    measure_state: callable | None,
) -> torch.Tensor | None:
    if measure_state is None or not isinstance(output, torch.Tensor) or not torch.is_complex(output):
        return None

    state = output.unsqueeze(0) if output.dim() == 1 else output
    if state.dim() != 2:
        return None

    measured = measure_state(state)
    if isinstance(measured, torch.Tensor):
        return measured
    return torch.as_tensor(measured)
