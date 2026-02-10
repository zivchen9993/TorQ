# torq/__init__.py
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("torq")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

# keep submodules visible for namespaced imports
from . import Ops, SingleQubitGates, Controls, Templates, Measure  # noqa: F401

# ---- explicit, flat re-exports ----
# Templates
from .Templates import (
    angle_embedding,
    cross_mesh_single_layer,
    basic_or_strongly_single_layer,
    global_entanglement_calc,
    get_initial_state,
    get_angle_embedding_sigmas,
    data_reuploading,
    data_reuploading_gates,
)
# Ops
from .Ops import (
    apply_matrix,
    multi_dim_tensor_product,
    multi_dim_matmul_reversed,
    kron_with_replace,
)
# Measure
from .Measure import (measure, measure_local, measure_local_Z)
# Single-qubit gates
from .SingleQubitGates import (
    ID1_like, sigma_X_like, sigma_Y_like, sigma_Z_like,
    ketbra00_like, ketbra01_like, ketbra10_like, ketbra11_like,
)
from .Rotations import (
    get_rx, get_ry, get_rz, get_rot_gate,
)
# Controls & Layout
from .Controls import (get_cnot_ops, get_single_two_qubit_gate)
from .Layout import (
    get_cnot_ladder,
    get_cross_mesh_control_gate_layer,
    get_single_qubit_pauli_rot_ops,
)
from .simple import Circuit, CircuitConfig

__all__ = [
    "__version__", "Ops", "SingleQubitGates", "Controls", "Templates", "Measure", "get_angle_embedding_sigmas",
    "angle_embedding", "cross_mesh_single_layer", "basic_or_strongly_single_layer",
    "global_entanglement_calc", "get_initial_state",
    "apply_matrix", "multi_dim_tensor_product", "multi_dim_matmul_reversed", "kron_with_replace",
    "measure", "measure_local", "measure_local_Z",
    "get_rx", "get_ry", "get_rz", "get_rot_gate",
    "ID1_like", "sigma_X_like", "sigma_Y_like", "sigma_Z_like",
    "ketbra00_like", "ketbra01_like", "ketbra10_like", "ketbra11_like",
    "get_cnot_ops", "get_single_two_qubit_gate",
    "get_cnot_ladder", "get_cross_mesh_control_gate_layer", "get_single_qubit_pauli_rot_ops", "data_reuploading",
    "data_reuploading_gates", "Circuit", "CircuitConfig"
]
