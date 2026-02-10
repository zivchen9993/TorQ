# torq/Ansatz.py
import torch
import torq as aq

class BaseAnsatz:
    """Uniform interface: param_shape, layer_op(layer_idx, weights)->[2**n,2**n]."""
    def __init__(self, n_qubits: int, n_layers: int, device=None):
        self.n = n_qubits
        self.L = n_layers
        self.device = device
    # override in subclasses
    param_shape: tuple = ()  # TODO: the parameter_shape is "per_layer" every time, probably we can just hardcode that and remove this attribute.


    def layer_op(self, layer_idx: int, weights: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class StronglyEntangling(BaseAnsatz):
    param_shape = ("per_layer", (None, 3))  # resolved later to (n_qubits,3)

    def __init__(self, n_qubits, n_layers, device=None):
        super().__init__(n_qubits, n_layers, device)
        # one CNOT ladder reused for all layers
        # uses the likable path so dtype/device are inferred from a tensor argument
        dummy = torch.empty(1, device=device)  # TODO: is the dummy really needed? can we just call get_cnot_ladder with r=0 and no x argument?
        self.cnot = aq.get_cnot_ladder(n_qubits, r=0, x=dummy)  # precompute once

    def layer_op(self, layer_idx, weights):
        # weights: [n_qubits,3]
        return aq.basic_entangling_single_layer(self.n, weights, self.cnot).to(weights.device)


class StronglyEntanglingAllToAll(BaseAnsatz):
    param_shape = ("per_layer", (None, 3))

    def __init__(self, n_qubits, n_layers, device=None):
        super().__init__(n_qubits, n_layers, device)
        dummy = torch.empty(1, device=device)
        # one ladder per "r" value
        self.cnots = [
            aq.get_cnot_ladder(n_qubits, r=r, x=dummy)
            for r in range(n_layers)
        ]

    def layer_op(self, layer_idx, weights):
        return aq.strongly_entangling_single_layer(self.n, weights, self.cnots[layer_idx]).to(weights.device)


class CrossMesh(BaseAnsatz):
    # single-rot: [n_qubits**2], double-rot: [n_qubits + n_qubits**2]
    # cx_rot: [n_qubits,3]
    def __init__(self, n_qubits, n_layers, variant: str, device=None):
        super().__init__(n_qubits, n_layers, device)
        self.variant = variant
        dummy = torch.empty(1, device=device)
        if variant == "cross_mesh_cx_rot":
            self.cross = aq.get_cross_mesh_control_gate_layer(
                n_qubits, sigma=aq.sigma_X_like, weights=None, device=device
            )
            self.param_shape = ("per_layer", (n_qubits, 3))
        elif variant == "cross_mesh":
            self.param_shape = ("per_layer", (n_qubits**2,))
            self.cross = None
        elif variant == "cross_mesh_2_rots":
            self.param_shape = ("per_layer", (n_qubits + n_qubits**2,))
            self.cross = None
        else:
            raise ValueError("Unknown cross-mesh variant")

    def layer_op(self, layer_idx, weights):
        if self.variant == "cross_mesh_cx_rot":
            # Templates.cross_mesh_single_layer can take a precomputed layer
            return aq.cross_mesh_single_layer(self.n, weights,
                                              sigma_single_rot_first=aq.get_rot_gate,
                                              sigma_single_rot_second=None,
                                              cnot_layer_precomputed=self.cross).to(weights.device)
        else:
            return aq.cross_mesh_single_layer(self.n, weights, sigma_single_rot_first=aq.get_rx,
                                              sigma_single_rot_second=(None if self.variant=="cross_mesh" else aq.get_rz),
                                              cnot_layer_precomputed=None).to(weights.device)

class NoEntanglement(BaseAnsatz):
    param_shape = ("per_layer", (None, 3))

    def layer_op(self, layer_idx, weights):
        return aq.get_single_qubit_pauli_rot_ops(self.n, weights, sigma_func=aq.get_rot_gate).to(weights.device)


def make_ansatz(name: str, n_qubits: int, n_layers: int, device=None) -> BaseAnsatz:
    if name == "basic_entangling":
        return StronglyEntangling(n_qubits, n_layers, device)
    if name == "strongly_entangling":
        return StronglyEntanglingAllToAll(n_qubits, n_layers, device)
    if name in ("cross_mesh", "cross_mesh_2_rots", "cross_mesh_cx_rot"):
        return CrossMesh(n_qubits, n_layers, name, device)
    if name == "no_entanglement_ansatz":
        return NoEntanglement(n_qubits, n_layers, device)
    raise ValueError(f"Unknown ansatz: {name}")


