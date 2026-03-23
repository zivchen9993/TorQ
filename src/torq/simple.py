"""Student-facing API for TorQ.

Example with angle scaling:

    import torch
    from torq.simple import Circuit, CircuitConfig

    cfg = CircuitConfig(angle_scaling_method="scale", angle_scaling=torch.pi)
    circuit = Circuit(n_qubits=4, n_layers=2, config=cfg)
    y = circuit(torch.rand(8, 4))
"""

from dataclasses import dataclass
import torch
import torch.nn as nn

from .QLayer import QLayer
from .Templates import (
    angle_embedding,
    data_reuploading,
    data_reuploading_gates,
    get_initial_state,
)
from .Measure import measure
from .SingleQubitGates import sigma_Z_like


@dataclass
class CircuitConfig:
    data_reupload_every: int = 0
    angle_scaling_method: str = "none"
    angle_scaling: float | None = 1.0
    reparametrize_sin_cos: bool = False
    single_rotation_gate: str = "rx"
    tile_rotation_params: int = 3
    tile_sublayers: int = 1
    tile_cyclic: bool = False
    init_identity: bool = False
    init_ones: bool = False
    init_pi_half: bool = False
    noise_all_q_layers: bool = False
    pennylane_backend: bool = False
    pennylane_dev_name: str | None = None
    observables: object = None
    pauli_measurement_chunk_size: int = 8

    def __post_init__(self) -> None:
        if self.data_reupload_every < 0:
            raise ValueError("data_reupload_every must be >= 0.")

        methods = {"none", "scale", "scale_with_bias", "asin", "acos"}
        if self.angle_scaling_method not in methods:
            raise ValueError(
                f"angle_scaling_method must be one of {sorted(methods)}. "
                f"Got: {self.angle_scaling_method!r}"
            )

        rotation_gates = {"x", "y", "z", "rx", "ry", "rz"}
        if (
            not isinstance(self.single_rotation_gate, str)
            or self.single_rotation_gate.lower() not in rotation_gates
        ):
            raise ValueError(
                "single_rotation_gate must be one of "
                f"{sorted(rotation_gates)}. Got: {self.single_rotation_gate!r}"
            )
        if self.tile_rotation_params not in (1, 3):
            raise ValueError(
                "tile_rotation_params must be one of (1, 3). "
                f"Got: {self.tile_rotation_params!r}"
            )
        if self.tile_sublayers < 1:
            raise ValueError("tile_sublayers must be >= 1.")

        if self.observables is not None:
            if isinstance(self.observables, (list, tuple)) and len(self.observables) == 0:
                raise ValueError("observables cannot be an empty list/tuple.")
        if self.pauli_measurement_chunk_size < 1:
            raise ValueError("pauli_measurement_chunk_size must be >= 1.")


class Circuit(nn.Module):
    """Student-friendly wrapper around QLayer."""

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        ansatz_name: str = "basic_entangling",
        config: CircuitConfig | None = None,
        weights: torch.Tensor | None = None,
        weights_last_layer_data_re: torch.Tensor | None = None,
        basis_angle_embedding: str = "X",
    ) -> None:
        super().__init__()
        self.config = config if config is not None else CircuitConfig()
        self.layer = QLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            ansatz_name=ansatz_name,
            config=self.config,
            weights=weights,
            weights_last_layer_data_re=weights_last_layer_data_re,
            basis_angle_embedding=basis_angle_embedding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

    @property
    def params(self) -> torch.nn.Parameter:
        return self.layer.params

    @property
    def params_last_layer_reupload(self) -> torch.nn.Parameter | None:
        return getattr(self.layer, "params_last_layer_reupload", None)


__all__ = [
    "CircuitConfig",
    "Circuit",
    "angle_embedding",
    "data_reuploading",
    "data_reuploading_gates",
    "get_initial_state",
    "measure",
    "sigma_Z_like",
]
