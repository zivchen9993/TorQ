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
    # Backward-compatible aliases for angle_scaling_method.
    angle_asin: bool = False
    angle_acos: bool = False
    scale_with_bias: bool = False
    reparametrize_sin_cos: bool = False
    init_identity: bool = False
    init_ones: bool = False
    init_pi_half: bool = False
    noise_all_q_layers: bool = False
    pennylane_backend: bool = False
    pennylane_dev_name: str | None = None
    local_observable_name: str = "Z"
    custom_local_observable: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.data_reupload_every < 0:
            raise ValueError("data_reupload_every must be >= 0.")

        methods = {"none", "scale", "scale_with_bias", "asin", "acos"}
        if self.angle_scaling_method not in methods:
            raise ValueError(
                f"angle_scaling_method must be one of {sorted(methods)}. "
                f"Got: {self.angle_scaling_method!r}"
            )

        legacy_methods = []
        if self.angle_asin:
            legacy_methods.append("asin")
        if self.angle_acos:
            legacy_methods.append("acos")
        if self.scale_with_bias:
            legacy_methods.append("scale_with_bias")

        if len(legacy_methods) > 1:
            raise ValueError(
                "Only one of angle_asin/angle_acos/scale_with_bias may be True."
            )
        if legacy_methods:
            if self.angle_scaling_method != "none":
                raise ValueError(
                    "Use either angle_scaling_method or legacy flags "
                    "(angle_asin/angle_acos/scale_with_bias), not both."
                )
            self.angle_scaling_method = legacy_methods[0]

        obs_name = self.local_observable_name.lower()
        allowed_names = {
            "z", "pauliz", "pauli_z", "sigmaz", "sigma_z",
            "x", "paulix", "pauli_x", "sigmax", "sigma_x",
            "y", "pauliy", "pauli_y", "sigmay", "sigma_y",
            "custom", "custom_hermitian", "local",
        }
        if obs_name not in allowed_names:
            raise ValueError(
                f"local_observable_name must be one of {sorted(allowed_names)}. "
                f"Got: {self.local_observable_name!r}"
            )
        if obs_name in {"custom", "custom_hermitian", "local"} and self.custom_local_observable is None:
            raise ValueError(
                "custom_local_observable must be provided when local_observable_name is custom."
            )


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
