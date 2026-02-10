from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from importlib.util import find_spec

import torch

from .Templates import get_angle_embedding_sigmas


@dataclass
class _PennyLaneRunner:
    circuit: callable
    angle_scaling_method: str
    angle_scaling: float
    basis_angle_embedding: str

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, scaled, _, _ = get_angle_embedding_sigmas(
            x,
            angle_scaling_method=self.angle_scaling_method,
            angle_scaling=self.angle_scaling,
            basis=self.basis_angle_embedding,
        )
        return torch.stack(self.circuit(scaled), dim=1).to(torch.float32)


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
    if getattr(layer, "basis_angle_embedding", "X").upper() not in ("X", "RX"):
        warnings.warn(
            "pennylane_backend=True but basis_angle_embedding is not 'X'; "
            "using TorQ backend.",
            RuntimeWarning,
        )
        return None

    comparison_cls = _load_pennylane_comparison_class()
    if comparison_cls is None:
        return None

    dev_name = getattr(layer.config, "pennylane_dev_name", "default.qubit") or "default.qubit"
    backend = comparison_cls(
        n_qubits=layer.n_qubits,
        n_layers=layer.n_layers,
        weights=layer.params,
        weights_last_layer_data_re=getattr(layer, "params_last_layer_reupload", None),
        data_reupload_every=layer.data_reupload_every,
        pennylane_dev_name=dev_name,
    )

    circuit = _select_pennylane_circuit(backend, layer.ansatz_name, layer.data_reupload_every)
    if circuit is None:
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


def _select_pennylane_circuit(pennylane_backend, ansatz_name: str, data_reupload_every: int):
    if data_reupload_every:
        match ansatz_name:
            case "basic_entangling":
                return pennylane_backend.data_re_circuit_strongly_entangling()
            case "strongly_entangling":
                return pennylane_backend.data_re_circuit_strongly_entangling_all_to_all()
            case "cross_mesh":
                return pennylane_backend.data_re_circuit_cross_mesh()
            case "cross_mesh_2_rots":
                return pennylane_backend.data_re_circuit_cross_mesh_2_rots()
            case "cross_mesh_cx_rot":
                return pennylane_backend.data_re_circuit_cross_mesh_cx_rot()
            case _:
                return None
    match ansatz_name:
        case "basic_entangling":
            return pennylane_backend.circuit_strongly_entangling()
        case "strongly_entangling":
            return pennylane_backend.circuit_strongly_entangling_all_to_all()
        case "cross_mesh":
            return pennylane_backend.circuit_cross_mesh()
        case "cross_mesh_2_rots":
            return pennylane_backend.circuit_cross_mesh_2_rots()
        case "cross_mesh_cx_rot":
            return pennylane_backend.circuit_cross_mesh_cx_rot()
        case _:
            return None
