"""Student-facing API for TorQ.

Example with angle scaling:

    import torch
    from .simple import Circuit, CircuitConfig

    cfg = CircuitConfig(angle_scaling=torch.pi, scale_with_bias=False)
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
    angle_scaling: float | None = 1.0
    angle_asin: bool = False
    angle_acos: bool = False
    scale_with_bias: bool = False
    reparametrize_sin_cos: bool = False
    init_identity: bool = False
    init_ones: bool = False
    init_pi_half: bool = False
    noise_all_q_layers: bool = False


class Circuit(nn.Module):
    """Student-friendly wrapper around QLayer."""

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        ansatz_name: str = "strongly_entangling",
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
