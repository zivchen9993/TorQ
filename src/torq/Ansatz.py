# torq/Ansatz.py
import torch

from . import Layout as layout, Rotations as rotations, SingleQubitGates as single, Templates as templates


def _resolve_single_rotation_gate(name: str):
    normalized = name.lower()
    rotation_map = {
        "x": rotations.get_rx,
        "rx": rotations.get_rx,
        "y": rotations.get_ry,
        "ry": rotations.get_ry,
        "z": rotations.get_rz,
        "rz": rotations.get_rz,
    }
    try:
        return rotation_map[normalized]
    except KeyError as exc:
        raise ValueError(
            "single_rotation_gate must be one of ('rx', 'ry', 'rz'). "
            f"Got: {name!r}"
        ) from exc


class BaseAnsatz:
    """Uniform interface: per_layer_param_shape, layer_op(layer_idx, weights)->[2**n,2**n]."""
    def __init__(self, n_qubits: int, n_layers: int, device=None):
        self.n = n_qubits
        self.L = n_layers
        self.device = device
    # override in subclasses
    per_layer_param_shape: tuple = ()

    def layer_op(self, layer_idx: int, weights: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BasicEntangling(BaseAnsatz):
    per_layer_param_shape = (None, 3)  # resolved later to (n_qubits, 3)

    def __init__(self, n_qubits, n_layers, device=None):
        super().__init__(n_qubits, n_layers, device)
        self._cnot_cache = {}

    def _get_cnot(self, weights: torch.Tensor) -> torch.Tensor:
        key = (weights.device.type, weights.device.index)
        cached = self._cnot_cache.get(key)
        if cached is None:
            cached = layout.get_cnot_ladder(self.n, r=0, x=weights)
            self._cnot_cache[key] = cached
        return cached

    def layer_op(self, layer_idx, weights):
        # weights: [n_qubits,3]
        return templates.basic_or_strongly_single_layer(self.n, weights, self._get_cnot(weights)).to(weights.device)


class SingleRotBasicEnt(BaseAnsatz):
    per_layer_param_shape = (None,)

    def __init__(self, n_qubits, n_layers, rotation_gate: str = "rx", device=None):
        super().__init__(n_qubits, n_layers, device)
        self._cnot_cache = {}
        self._single_rot = _resolve_single_rotation_gate(rotation_gate)

    def _get_cnot(self, weights: torch.Tensor) -> torch.Tensor:
        key = (weights.device.type, weights.device.index)
        cached = self._cnot_cache.get(key)
        if cached is None:
            cached = layout.get_cnot_ladder(self.n, r=0, x=weights)
            self._cnot_cache[key] = cached
        return cached

    def layer_op(self, layer_idx, weights):
        return templates.basic_or_strongly_single_layer(
            self.n,
            weights,
            self._get_cnot(weights),
            sigma_single_rot=self._single_rot,
        ).to(weights.device)


class StronglyEntangling(BaseAnsatz):
    per_layer_param_shape = (None, 3)

    def __init__(self, n_qubits, n_layers, device=None):
        super().__init__(n_qubits, n_layers, device)
        self._cnot_cache = {}

    def _get_cnot(self, layer_idx: int, weights: torch.Tensor) -> torch.Tensor:
        key = (layer_idx, weights.device.type, weights.device.index)
        cached = self._cnot_cache.get(key)
        if cached is None:
            cached = layout.get_cnot_ladder(self.n, r=layer_idx, x=weights)
            self._cnot_cache[key] = cached
        return cached

    def layer_op(self, layer_idx, weights):
        return templates.basic_or_strongly_single_layer(self.n, weights, self._get_cnot(layer_idx, weights)).to(weights.device)


class CrossMesh(BaseAnsatz):
    # single-rot: [n_qubits**2], double-rot: [n_qubits + n_qubits**2]
    # cx_rot: [n_qubits,3]
    def __init__(self, n_qubits, n_layers, variant: str, device=None):
        super().__init__(n_qubits, n_layers, device)
        self.variant = variant
        self._cross_cache = {}
        if variant == "cross_mesh_cx_rot":
            self.per_layer_param_shape = (n_qubits, 3)
        elif variant == "cross_mesh":
            self.per_layer_param_shape = (n_qubits**2,)
        elif variant == "cross_mesh_2_rots":
            self.per_layer_param_shape = (n_qubits + n_qubits**2,)
        else:
            raise ValueError("Unknown cross-mesh variant")

    def _get_cross_layer(self, weights: torch.Tensor) -> torch.Tensor:
        key = (weights.device.type, weights.device.index)
        cached = self._cross_cache.get(key)
        if cached is None:
            cached = layout.get_cross_mesh_control_gate_layer(
                self.n,
                sigma=single.sigma_X_like,
                weights=None,
                device=weights.device,
            )
            self._cross_cache[key] = cached
        return cached

    def layer_op(self, layer_idx, weights):
        if self.variant == "cross_mesh_cx_rot":
            # Templates.cross_mesh_single_layer can take a precomputed layer
            return templates.cross_mesh_single_layer(
                self.n,
                weights,
                sigma_single_rot_first=rotations.get_rot_gate,
                sigma_single_rot_second=None,
                cnot_layer_precomputed=self._get_cross_layer(weights),
            ).to(weights.device)
        else:
            return templates.cross_mesh_single_layer(
                self.n,
                weights,
                sigma_single_rot_first=rotations.get_rx,
                sigma_single_rot_second=(None if self.variant == "cross_mesh" else rotations.get_rz),
                cnot_layer_precomputed=None,
            ).to(weights.device)


class Tile(BaseAnsatz):
    def __init__(
        self,
        n_qubits,
        n_layers,
        rotation_params: int = 3,
        single_rotation_gate: str = "rx",
        n_sublayers: int = 1,
        cyclic: bool = False,
        device=None,
    ):
        super().__init__(n_qubits, n_layers, device)
        self.rotation_params = rotation_params
        self.n_sublayers = n_sublayers
        self.cyclic = cyclic
        self._cnot_cache = {}

        if rotation_params == 3:
            self.per_layer_param_shape = (None, 3)
            self._single_rot = rotations.get_rot_gate
        elif rotation_params == 1:
            self.per_layer_param_shape = (None,)
            self._single_rot = _resolve_single_rotation_gate(single_rotation_gate)
        else:
            raise ValueError(
                "tile_rotation_params must be one of (1, 3). "
                f"Got: {rotation_params!r}"
            )

    def _get_cnot(self, weights: torch.Tensor) -> torch.Tensor:
        key = (weights.device.type, weights.device.index)
        cached = self._cnot_cache.get(key)
        if cached is None:
            cached = layout.get_cnot_brick_wall(
                self.n,
                n_sublayers=self.n_sublayers,
                cyclic=self.cyclic,
                x=weights,
            )
            self._cnot_cache[key] = cached
        return cached

    def layer_op(self, layer_idx, weights):
        return templates.basic_or_strongly_single_layer(
            self.n,
            weights,
            self._get_cnot(weights),
            sigma_single_rot=self._single_rot,
        ).to(weights.device)


class NoEntanglement(BaseAnsatz):
    per_layer_param_shape = (None, 3)

    def layer_op(self, layer_idx, weights):
        return layout.get_single_qubit_pauli_rot_ops(self.n, weights, sigma_func=rotations.get_rot_gate).to(weights.device)


def make_ansatz(name: str, n_qubits: int, n_layers: int, device=None, config=None) -> BaseAnsatz:
    if name == "basic_entangling":
        return BasicEntangling(n_qubits, n_layers, device)
    if name == "single_rot_basic_ent":
        rotation_gate = getattr(config, "single_rotation_gate", "rx")
        return SingleRotBasicEnt(n_qubits, n_layers, rotation_gate=rotation_gate, device=device)
    if name == "strongly_entangling":
        return StronglyEntangling(n_qubits, n_layers, device)
    if name in ("cross_mesh", "cross_mesh_2_rots", "cross_mesh_cx_rot"):
        return CrossMesh(n_qubits, n_layers, name, device)
    if name == "tile":
        return Tile(
            n_qubits,
            n_layers,
            rotation_params=getattr(config, "tile_rotation_params", 3),
            single_rotation_gate=getattr(config, "single_rotation_gate", "rx"),
            n_sublayers=getattr(config, "tile_sublayers", 1),
            cyclic=getattr(config, "tile_cyclic", False),
            device=device,
        )
    if name == "no_entanglement_ansatz":
        return NoEntanglement(n_qubits, n_layers, device)
    raise ValueError(f"Unknown ansatz: {name}")
