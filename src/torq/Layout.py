import torch
from . import Controls as controls, Ops as ops, Rotations as rotations
from ._like import likeable


def get_single_qubit_pauli_rot_ops(n_qubits, theta, sigma_func=rotations.get_rz):
    """ Constructs the full 2^n_qubits x 2^n_qubits operator as the tensor
    (Kronecker) product of single-qubit rotations computed in a fully vectorized manner.

    For scalar rotations (e.g. using get_rz), theta is expected to be a 1D tensor of shape
    [n_qubits]. For composite rotations (e.g. using get_rot_gate), theta should have shape
    [n_qubits, 3] where each row corresponds to the three parameters for that qubit.

    Args:
      n_qubits (int): Number of qubits.
      theta (torch.Tensor): Rotation angles. Either shape [n_qubits] or [n_qubits, num_params].
      sigma_func (function): A function that returns a 2x2 matrix given an angle (or vector of
                             angles). It must support batched input: if theta has shape [n_qubits, ...],
                             then sigma_func(theta) must return a tensor of shape [n_qubits, 2, 2].

    Returns:
      torch.Tensor: The full operator (2^n_qubits x 2^n_qubits).
    """
    # Ensure theta is at least 2D so that each row corresponds to one qubit.
    if theta.dim() == 1:
        # In the scalar case, add a trailing dimension so that each row is [angle]
        theta = theta.unsqueeze(1)  # Now shape: [n_qubits, 1]

    # Call sigma_func once with all the angles.
    # The expectation is that sigma_func supports batched input and returns [n_qubits, 2, 2].
    gates = sigma_func(theta)  # e.g. using get_rz: if theta was [n_qubits,1] then output [n_qubits,2,2]

    # Now, compute the full operator via the Kronecker product.
    # We create a list of the per-qubit 2x2 gates.
    gates_list = [gates[i] for i in range(n_qubits)]

    # Use your optimized multi_dim_tensor_product (which uses slicing/einsum) to get the tensor product.
    return ops.multi_dim_tensor_product(*gates_list)


@likeable
def get_cnot_ladder(n_qubits, r=0, *, dtype, device):
    gate_items_list = []
    if (r + 1) == n_qubits:
        return torch.eye(2 ** n_qubits, dtype=dtype, device=device).unsqueeze(0)
    for control_qubit_id in range(n_qubits):
        target_qubit_id = (control_qubit_id + r + 1) % n_qubits
        current_CNOT = controls.get_cnot_ops(n_qubits, control_qubit_id, target_qubit_id, device=device)  # in the loop, r starts with 0
        current_CNOT = current_CNOT.unsqueeze(0)  # add batch dimension
        gate_items_list.append(current_CNOT)
    return ops.multi_dim_matmul_reversed(*gate_items_list)


def get_cross_mesh_control_gate_layer(n_qubits, sigma, device, weights=None):
    # Instead of nested loops with repeated list constructions, build a list
    # of controlled gate operators using a list comprehension over the valid pairs.
    # We assume that the weight parameters for controlled operations are supplied in a
    # fixed order.
    control_target_pairs = [(c, t) for c in reversed(range(n_qubits))
                            for t in reversed(range(n_qubits)) if c != t]
    controlled_ops = []
    for wi, (control, target) in enumerate(control_target_pairs):
        # Get the control-target gate using the existing function.
        # This is essentially:
        #   op = kron_with_replace(n_qubits, {control: ketbra00}, device)
        #      + kron_with_replace(n_qubits, {control: ketbra11, target: sigma(weights[wi])}, device)
        # Here we can call get_single_control_gate_ops to get this operator.
        phi = weights[wi] if weights is not None else None
        op = controls.get_single_two_qubit_gate(n_qubits, control_qubit_id=control,
                                       target_qubit_id=target, phi=phi,
                                       sigma=sigma, device=device)
        controlled_ops.append(op.unsqueeze(0))
    # Multiply all controlled operations together using the optimized reversed multiplication.
    return ops.multi_dim_matmul_reversed(*controlled_ops)

