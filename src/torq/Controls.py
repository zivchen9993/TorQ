import torch
from . import SingleQubitGates as single, Ops as ops, Rotations as rotations
from ._like import likeable


@likeable
def get_cnot_ops(n_qubits, control_qubit_id, target_qubit_id, *, dtype, device):  # old and working
    # Create device-specific constants
    ID = single.ID1_like(dtype=dtype, device=device)
    ketbra00 = single.ketbra00_like(dtype=dtype, device=device)
    ketbra11 = single.ketbra11_like(dtype=dtype, device=device)
    sigma_X = single.sigma_X_like(dtype=dtype, device=device)

    if control_qubit_id < target_qubit_id:
        op1 = ops.multi_dim_tensor_product(*([ID] * control_qubit_id), ketbra00,
                                       *([ID] * (n_qubits - control_qubit_id - 1)))
        op2 = ops.multi_dim_tensor_product(*([ID] * control_qubit_id), ketbra11,
                                       *([ID] * (target_qubit_id - control_qubit_id - 1)),
                                       sigma_X,
                                       *([ID] * (n_qubits - target_qubit_id - 1)))
        return op1 + op2
    else:
        op1 = ops.multi_dim_tensor_product(*([ID] * control_qubit_id), ketbra00,
                                       *([ID] * (n_qubits - control_qubit_id - 1)))
        op2 = ops.multi_dim_tensor_product(*([ID] * target_qubit_id), sigma_X,
                                       *([ID] * (control_qubit_id - target_qubit_id - 1)),
                                       ketbra11,
                                       *([ID] * (n_qubits - control_qubit_id - 1)))
        return op1 + op2


@likeable
def get_single_two_qubit_gate(n_qubits, control_qubit_id, target_qubit_id, *, dtype, device, phi=None, sigma=None):
    """
    Constructs the controlled gate acting on a specific control–target qubit pair.

    It implements:
      op = kron(I,...,ketbra00,...,I)  +  kron(I,...,ketbra11,...,sigma(phi),...,I)
    where the replacements occur at:
      - The control qubit: ketbra00 or ketbra11
      - The target qubit (in the second term): sigma(phi) acting on the target qubit.

    Uses kron_with_replace to insert the non-identity operators.

    Args:
      n_qubits (int): Total number of qubits.
      control_qubit_id (int): Index of the control qubit.
      target_qubit_id (int): Index of the target qubit.
      phi (torch.Tensor or scalar): Parameter(s) that define the target rotation.
      sigma (function): Function that returns a 2×2 rotation gate given phi.
                        Expected to return a tensor of shape [1,2,2] (which is squeezed to 2×2).
      device (torch.device, optional): Device to use.

    Returns:
      torch.Tensor: The full operator (2^n_qubits x 2^n_qubits) for this controlled gate.
    """
    if sigma is None:
        sigma = rotations.get_rz
    # Get device-specific constants:
    ketbra00 = single.ketbra00_like(dtype=dtype, device=device)
    ketbra11 = single.ketbra11_like(dtype=dtype, device=device)
    # Compute the single-qubit rotation to apply at the target qubit.
    # We expect sigma(phi) to return a tensor with shape [1,2,2]; squeeze removes the batch.
    if phi is None:
        op_gate = single.sigma_X_like(dtype=dtype, device=device)
    else:
        op_gate = sigma(phi).squeeze(0)
    # For both cases, the branch with control in state |0> is the same:
    op0 = ops.kron_with_replace(n_qubits, {control_qubit_id: ketbra00}, dtype=dtype, device=device)

    # For the branch with control in state |1>, we also replace the target qubit.
    if control_qubit_id < target_qubit_id:
        op1 = ops.kron_with_replace(n_qubits, {control_qubit_id: ketbra11, target_qubit_id: op_gate}, dtype=dtype, device=device)
    else:
        op1 = ops.kron_with_replace(n_qubits, {target_qubit_id: op_gate, control_qubit_id: ketbra11}, dtype=dtype, device=device)

    return op0 + op1
