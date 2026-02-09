import torch
import math
from . import Ops as ops, Layout as layout, Rotations as rotations
from ._like import likeable


@likeable
def get_initial_state(n_qubits, batch_size=1, *, dtype, device):
    state = torch.zeros(batch_size, 2 ** n_qubits, 1, dtype=dtype, device=device)
    state[:, 0, 0] = 1.0
    return state


def get_angle_embedding_sigmas(angles, angle_scaling_method='none', angle_scaling=torch.pi, basis='X'):  # angle scaling methods: none, 'scale' (without bias), 'scale_with_bias', 'asin', 'acos'
    """
    Important note: when using asin or acos, it clamps the angles so they won't get to -1 or 1 and blow up the derivatives
    """
    # clamping to avoid blowup in the derivatives in asin and acos scaling methods (d_asin/dx = 1/sqrt(1-x^2);  d_acos/dx = -d_asin/dx = -1/sqrt(1-x^2))
    eps = 1e-7  # tiny shrink, enough to keep derivative finite

    match angle_scaling_method:
        case 'scale':  # scale without bias
            angles = angles * angle_scaling
        case 'scale_with_bias':
            angles = (angles + 1) * (angle_scaling / 2)  # scale to [0, scaling] - which means [0, pi]
        case 'asin':
            angles = angles.clamp(min=-1.0 + eps, max=1.0 - eps)
            angles = torch.asin(angles) + (torch.pi / 2)
        case 'acos':
            angles = angles.clamp(min=-1.0 + eps, max=1.0 - eps)
            angles = torch.acos(angles)
        case 'none':
            pass
        case _:
            raise ValueError(f"Unknown scaling_method: {angle_scaling_method}")
        
    # if scaling_method == 'acos' or scaling_method == 'asin':
    #     eps = 1e-7  # tiny shrink, enough to keep derivative finite
    #     angles = angles.clamp(min=-1.0 + eps, max=1.0 - eps)
    #     if scaling_method == 'acos':
    #         angles = torch.acos(angles)
    #     else:
    #         angles = torch.asin(angles) + (torch.pi / 2)
    # elif scale is not None:  # can't work with both scaling and arcsin
    #     if scale_with_bias:
    #         angles = (angles + 1) * (scale / 2)  # scale to [0, scale] - which means [0, pi]
    #     else:
    #         angles = angles * scale
    if angles.dim() == 1:
        angles = angles.unsqueeze(0)
    B, n_qubits = angles.shape

    # Choose the appropriate rotation function.
    if basis.upper() in ['X', 'RX']:
        sigma = rotations.get_rx  # Must also support the parameter only_first_column.
    elif basis.upper() in ['Y', 'RY']:
        sigma = rotations.get_ry
    elif basis.upper() in ['Z', 'RZ']:
        sigma = rotations.get_rz
    else:
        raise ValueError("Basis must be 'X', 'Y', or 'Z'")
    return sigma, angles, B, n_qubits


def angle_embedding(angles, angle_scaling_method='none', angle_scaling=torch.pi, basis='X'):
    """ Create angle embedding by computing all single-qubit gates in parallel.
    If the state is the default initial state (|0...0⟩),
    then only the first column of each gate is computed (using only_first_column)
    to avoid the cost of building the full 2x2 operator.

    angles: Tensor of shape [B, n_qubits] or [n_qubits] (if unbatched)
    state: Tensor of shape [B, 2**n_qubits, 1]
    scale: Optional factor to scale the angles.
    Basis: 'X' (or 'RX'), 'Y' (or 'RY'), or 'Z' (or 'RZ')

    Returns: New state after angle embedding, of shape [B, 2**n_qubits, 1]
    """
    sigma, angles, B, n_qubits = get_angle_embedding_sigmas(angles, angle_scaling_method, angle_scaling, basis)

    # --- Optimized default-case ---
    # Flatten angles: shape [B * n_qubits]
    angles_flat = angles.reshape(-1)
    # Call sigma with only_first_column=True so that each call returns shape [B*n_qubits, 2, 1]
    vectors_flat = sigma(angles_flat, only_first_column=True)
    # Reshape to [B, n_qubits, 2, 1]
    vectors = vectors_flat.reshape(B, n_qubits, 2, 1)
    # Unbind the n_qubits dimension without an explicit Python loop.
    vectors_list = vectors.transpose(0, 1).unbind(0)  # list of length n_qubits, each [B, 2, 1]
    # Compute the batched Kronecker product to get the final state.
    new_state = ops.multi_dim_tensor_product(*vectors_list)
    return new_state


def data_reuploading(state, gates, qubit_order="msb"):
    B, n_qubits = gates.shape[:2]
    dim = 2 ** n_qubits
    if state.dim() == 1 and state.numel() == dim:
        state = state.view(1, dim, 1).expand(B, -1, -1).contiguous()
    elif state.dim() == 2 and state.shape == (dim, 1):
        state = state.unsqueeze(0).expand(B, -1, -1).contiguous()
    elif state.dim() == 2 and state.shape[1] == dim:
        state = state.unsqueeze(-1)
    elif not (state.dim() == 3 and state.shape[:2] == (B, dim)):
        raise ValueError("state must be [B, 2**n] or [B, 2**n, 1]")

    # ← call the fast state-vector update
    state = ops.apply_single_qubit_wall_batched(state, gates, n_qubits, qubit_order=qubit_order)
    return state


def data_reuploading_gates(angles, angle_scaling_method='none', angle_scaling=torch.pi, basis='X'):
    sigma, angles, B, n_qubits = get_angle_embedding_sigmas(angles, angle_scaling_method, angle_scaling, basis)

    angles_flat = angles.reshape(-1)
    gates_flat = sigma(angles_flat, only_first_column=False)  # [B*n,2,2]
    gates = gates_flat.reshape(B, n_qubits, 2, 2)  # [B,n,2,2]

    return gates


def cross_mesh_single_layer(n_qubits, weights,
                            sigma_single_rot_first=rotations.get_rx,
                            sigma_single_rot_second=rotations.get_rz,
                            sigma_cross=rotations.get_rz,
                            cnot_layer_precomputed=None):
    """
    Creates a single cross-mesh layer.
    For the double–rot case the first n_qubits parameters correspond to the first rotation
    and the next n_qubits to the second rotation. They are combined per qubit.

    Args:
      weights: tensor of shape either [n_qubits^2] (for single rot) or
               [n_qubits + n_qubits^2] (for double rot).
    """
    if weights.shape[0] == (n_qubits + n_qubits ** 2):  # double rot case
        # Separate the two sets of rotation parameters.
        theta_first = weights[:n_qubits]  # shape: [n_qubits, ...]
        theta_second = weights[n_qubits:2 * n_qubits]  # same shape
        # Compute the per–qubit gates (using vectorized sigma functions)
        # Here, we assume that sigma_single_rot_first and sigma_single_rot_second
        # support batched input: for example, if each gate is defined by three parameters,
        # then theta_first and theta_second have shape [n_qubits, 3] and the sigma functions
        # return [n_qubits, 2, 2].
        gates_first = sigma_single_rot_first(theta_first)  # [n_qubits, 2, 2]
        gates_second = sigma_single_rot_second(theta_second)  # [n_qubits, 2, 2]
        # Combine the rotations per qubit. (The order depends on your decomposition.)
        # For instance, if you wish to apply the first rotation and then the second:
        combined_gates = torch.matmul(gates_second, gates_first)  # [n_qubits, 2, 2]
        # Build one combined rotation wall.
        rot_wall = ops.multi_dim_tensor_product(*[combined_gates[i] for i in range(n_qubits)])
        # Use the remaining weights for the cross-mesh control gates.
        cross_weights = weights[2 * n_qubits:]
    elif weights.shape[0] == (n_qubits ** 2):  # single rot case
        # Single rotation version: use the first n_qubits parameters as usual.
        rot_wall = layout.get_single_qubit_pauli_rot_ops(n_qubits, weights[:n_qubits],
                                                  sigma_func=sigma_single_rot_first)
        cross_weights = weights[n_qubits:]
    else:  # cx-rot case
        rot_wall = layout.get_single_qubit_pauli_rot_ops(n_qubits, weights, sigma_func=rotations.get_rot_gate)
        cross_mesh_layer = cnot_layer_precomputed
        return ops.multi_dim_matmul_reversed(rot_wall, cross_mesh_layer)

    cross_mesh_layer = layout.get_cross_mesh_control_gate_layer(n_qubits, sigma=sigma_cross,
                                                         weights=cross_weights, device=weights.device)
    # Compose the rotation wall and the control part.
    return ops.multi_dim_matmul_reversed(rot_wall, cross_mesh_layer)


def strongly_entangling_single_layer(n_qubits, weights, cnot_layer=None):
    """
        Creates a single strongly entangling layer.
        weights: tensor of shape [n_qubits, 3] (shared for all batch elements)
        Returns: operator of shape [2**n_qubits, 2**n_qubits]
        """
    layer_ops = []
    # Build the rotation wall using the weight parameters.
    rot_wall = layout.get_single_qubit_pauli_rot_ops(n_qubits, weights, sigma_func=rotations.get_rot_gate)
    layer_ops.append(rot_wall)
    layer_ops.append(cnot_layer)
    return ops.multi_dim_matmul_reversed(*layer_ops)


def global_entanglement_calc(psi: torch.Tensor):
    """
    psi: [B, 2**n]
    returns: (Q_lin, Q_R2) both as scalar tensors
    """
    B, dim = psi.shape
    n = int(math.log2(dim))
    D = dim // 2

    # reshape + build M: [B, n, 2, D]
    psi_t = psi.view(B, *([2]*n))
    Ms = []
    for k in range(n):
        perm = [0, k+1] + [i for i in range(1, n+1) if i!=k+1]
        Ms.append(psi_t.permute(*perm).reshape(B, 2, D))
    M = torch.stack(Ms, dim=1)   # [B,n,2,D]

    # partial trace → purity [B,n]
    rho    = torch.einsum('b n i d, b n j d -> b n i j', M, M.conj())
    purity = rho.reshape(B, n, 4).abs().pow(2).sum(dim=2)

    # Meyer–Wallach batch average:
    mean_purity = purity.mean()            # ⟨Trρ²⟩ over b,k
    Q_lin = 2 * (1 - mean_purity)          # scalar

    # Rényi-2 batch average:
    mean_logp = torch.log(purity).mean()   # ⟨log Trρ²⟩ over b,k
    Q_R2 = -mean_logp                      # scalar

    return Q_lin, Q_R2
