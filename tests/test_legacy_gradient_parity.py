import pytest

torch = pytest.importorskip("torch")

import torq as tq
from torq.Ansatz import make_ansatz as make_torq_ansatz
from torq.simple import Circuit, CircuitConfig

from _legacy_utils import (
    legacy_basic_entangling_forward,
    legacy_scaling_kwargs,
    legacy_strongly_entangling_all_to_all_forward,
    load_legacy_ansatz_module,
    load_legacy_quantum_lib,
)


def _weighted_real_scalar(tensor: torch.Tensor) -> torch.Tensor:
    real_part = tensor.real if torch.is_complex(tensor) else tensor
    coeffs = torch.linspace(
        0.5,
        1.5,
        real_part.numel(),
        dtype=real_part.dtype,
        device=real_part.device,
    ).reshape_as(real_part)
    loss = (real_part * coeffs).sum()
    if torch.is_complex(tensor):
        loss = loss + 0.37 * (tensor.imag * coeffs.flip(-1)).sum()
    return loss


def _assert_output_and_gradient_parity(
    legacy_output: torch.Tensor,
    torq_output: torch.Tensor,
    legacy_inputs,
    torq_inputs,
    *,
    atol: float,
    rtol: float,
) -> None:
    assert torch.allclose(legacy_output, torq_output, atol=atol, rtol=rtol)

    legacy_grads = torch.autograd.grad(_weighted_real_scalar(legacy_output), legacy_inputs)
    torq_grads = torch.autograd.grad(_weighted_real_scalar(torq_output), torq_inputs)

    for legacy_grad, torq_grad in zip(legacy_grads, torq_grads):
        assert torch.allclose(legacy_grad, torq_grad, atol=atol, rtol=rtol)


@pytest.fixture(scope="module")
def legacy_qg():
    return load_legacy_quantum_lib()


@pytest.fixture(scope="module")
def legacy_ansatz_module():
    return load_legacy_ansatz_module()


@pytest.mark.full
@pytest.mark.parametrize("angle_scaling_method", ["none", "scale", "scale_with_bias", "asin", "acos"])
@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_angle_embedding_gradients_match_legacy(legacy_qg, angle_scaling_method, basis):
    torch.manual_seed(101)
    x_legacy = (torch.rand(4, 3, dtype=torch.float64) * 1.8 - 0.9).requires_grad_(True)
    x_torq = x_legacy.detach().clone().requires_grad_(True)
    legacy_kwargs = legacy_scaling_kwargs(angle_scaling_method, torch.pi)

    legacy_out = legacy_qg.angle_embedding(x_legacy, basis=basis, **legacy_kwargs)
    torq_out = tq.angle_embedding(
        x_torq,
        angle_scaling_method=angle_scaling_method,
        angle_scaling=torch.pi,
        basis=basis,
    )

    _assert_output_and_gradient_parity(
        legacy_out,
        torq_out,
        (x_legacy,),
        (x_torq,),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.full
@pytest.mark.parametrize("angle_scaling_method", ["none", "scale", "scale_with_bias", "asin", "acos"])
def test_data_reuploading_gradients_match_legacy(legacy_qg, angle_scaling_method):
    torch.manual_seed(103)
    x_legacy = (torch.rand(3, 4, dtype=torch.float64) * 1.8 - 0.9).requires_grad_(True)
    x_torq = x_legacy.detach().clone().requires_grad_(True)
    legacy_kwargs = legacy_scaling_kwargs(angle_scaling_method, torch.pi)

    legacy_gates, legacy_batch_size, legacy_n_qubits = legacy_qg.data_reuploading_gates(
        x_legacy,
        basis="Y",
        **legacy_kwargs,
    )
    torq_gates = tq.data_reuploading_gates(
        x_torq,
        angle_scaling_method=angle_scaling_method,
        angle_scaling=torch.pi,
        basis="Y",
    )

    base_state = torch.complex(
        torch.randn(legacy_batch_size, 2 ** legacy_n_qubits, dtype=torch.float64),
        torch.randn(legacy_batch_size, 2 ** legacy_n_qubits, dtype=torch.float64),
    )
    legacy_state_leaf = (base_state / torch.linalg.norm(base_state, dim=1, keepdim=True)).detach().clone().requires_grad_(True)
    torq_state_leaf = legacy_state_leaf.detach().clone().requires_grad_(True)
    legacy_state = legacy_state_leaf * 1
    torq_state = torq_state_leaf * 1

    legacy_out = legacy_qg.data_reuploading(
        legacy_batch_size,
        legacy_n_qubits,
        legacy_state,
        legacy_gates,
    )
    torq_out = tq.data_reuploading(torq_state, torq_gates)

    _assert_output_and_gradient_parity(
        legacy_out,
        torq_out,
        (x_legacy, legacy_state_leaf),
        (x_torq, torq_state_leaf),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.full
def test_basic_entangling_layer_gradients_match_legacy(legacy_ansatz_module):
    torch.manual_seed(107)
    legacy_weights = torch.randn(4, 3, dtype=torch.float32, requires_grad=True)
    torq_weights = legacy_weights.detach().clone().requires_grad_(True)

    legacy_out = legacy_ansatz_module.make_ansatz("strongly_entangling", 4, 3).layer_op(2, legacy_weights)
    torq_out = make_torq_ansatz("basic_entangling", 4, 3).layer_op(2, torq_weights)

    _assert_output_and_gradient_parity(
        legacy_out,
        torq_out,
        (legacy_weights,),
        (torq_weights,),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.full
def test_strongly_entangling_layer_gradients_match_legacy(legacy_ansatz_module):
    torch.manual_seed(109)
    legacy_weights = torch.randn(4, 3, dtype=torch.float32, requires_grad=True)
    torq_weights = legacy_weights.detach().clone().requires_grad_(True)

    legacy_out = legacy_ansatz_module.StronglyEntanglingAllToAll(4, 3).layer_op(2, legacy_weights)
    torq_out = make_torq_ansatz("strongly_entangling", 4, 3).layer_op(2, torq_weights)

    _assert_output_and_gradient_parity(
        legacy_out,
        torq_out,
        (legacy_weights,),
        (torq_weights,),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.full
@pytest.mark.parametrize("ansatz_name", ["cross_mesh", "cross_mesh_2_rots", "cross_mesh_cx_rot", "no_entanglement_ansatz"])
def test_shared_ansatz_variant_gradients_match_legacy(legacy_ansatz_module, ansatz_name):
    torch.manual_seed(113)
    n_qubits = 3
    n_layers = 2

    if ansatz_name == "cross_mesh":
        legacy_weights = torch.randn(n_qubits ** 2, dtype=torch.float32, requires_grad=True)
    elif ansatz_name == "cross_mesh_2_rots":
        legacy_weights = torch.randn(n_qubits + n_qubits ** 2, dtype=torch.float32, requires_grad=True)
    else:
        legacy_weights = torch.randn(n_qubits, 3, dtype=torch.float32, requires_grad=True)
    torq_weights = legacy_weights.detach().clone().requires_grad_(True)

    legacy_out = legacy_ansatz_module.make_ansatz(ansatz_name, n_qubits, n_layers).layer_op(1, legacy_weights)
    torq_out = make_torq_ansatz(ansatz_name, n_qubits, n_layers).layer_op(1, torq_weights)

    _assert_output_and_gradient_parity(
        legacy_out,
        torq_out,
        (legacy_weights,),
        (torq_weights,),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.full
def test_basic_entangling_circuit_gradients_match_legacy_manual_forward(legacy_qg, legacy_ansatz_module):
    torch.manual_seed(127)
    legacy_weights = torch.randn(2, 1, 3, 3, dtype=torch.float32, requires_grad=True)
    torq_weights = legacy_weights.detach().clone().requires_grad_(True)
    x_legacy = (torch.rand(4, 3, dtype=torch.float32) * 1.8 - 0.9).requires_grad_(True)
    x_torq = x_legacy.detach().clone().requires_grad_(True)

    legacy_out = legacy_basic_entangling_forward(
        legacy_qg,
        legacy_ansatz_module,
        x_legacy,
        legacy_weights,
        angle_scaling_method="scale",
        angle_scaling=torch.pi,
        basis="X",
    )

    circuit = Circuit(
        n_qubits=3,
        n_layers=2,
        ansatz_name="basic_entangling",
        config=CircuitConfig(
            angle_scaling_method="scale",
            angle_scaling=torch.pi,
            basis_angle_embedding="X",
        ),
        weights=torq_weights,
    )
    torq_out = circuit(x_torq)

    _assert_output_and_gradient_parity(
        legacy_out,
        torq_out,
        (x_legacy, legacy_weights),
        (x_torq, circuit.params),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.full
def test_strongly_entangling_circuit_gradients_match_legacy_all_to_all_manual_forward(legacy_qg, legacy_ansatz_module):
    torch.manual_seed(131)
    legacy_weights = torch.randn(2, 1, 3, 3, dtype=torch.float32, requires_grad=True)
    torq_weights = legacy_weights.detach().clone().requires_grad_(True)
    x_legacy = (torch.rand(4, 3, dtype=torch.float32) * 1.8 - 0.9).requires_grad_(True)
    x_torq = x_legacy.detach().clone().requires_grad_(True)

    legacy_out = legacy_strongly_entangling_all_to_all_forward(
        legacy_qg,
        legacy_ansatz_module,
        x_legacy,
        legacy_weights,
        angle_scaling_method="scale",
        angle_scaling=torch.pi,
        basis="Z",
    )

    circuit = Circuit(
        n_qubits=3,
        n_layers=2,
        ansatz_name="strongly_entangling",
        config=CircuitConfig(
            angle_scaling_method="scale",
            angle_scaling=torch.pi,
            basis_angle_embedding="Z",
        ),
        weights=torq_weights,
    )
    torq_out = circuit(x_torq)

    _assert_output_and_gradient_parity(
        legacy_out,
        torq_out,
        (x_legacy, legacy_weights),
        (x_torq, circuit.params),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.full
def test_basic_entangling_data_reupload_circuit_gradients_match_legacy_manual_forward(legacy_qg, legacy_ansatz_module):
    torch.manual_seed(137)
    data_reupload_every = 2
    legacy_weights = torch.randn(2, data_reupload_every, 3, 3, dtype=torch.float32, requires_grad=True)
    torq_weights = legacy_weights.detach().clone().requires_grad_(True)
    legacy_last = torch.randn(data_reupload_every, 3, 3, dtype=torch.float32, requires_grad=True)
    torq_last = legacy_last.detach().clone().requires_grad_(True)
    x_legacy = (torch.rand(4, 3, dtype=torch.float32) * 1.8 - 0.9).requires_grad_(True)
    x_torq = x_legacy.detach().clone().requires_grad_(True)

    legacy_out = legacy_basic_entangling_forward(
        legacy_qg,
        legacy_ansatz_module,
        x_legacy,
        legacy_weights,
        angle_scaling_method="scale_with_bias",
        angle_scaling=torch.pi,
        basis="Y",
        data_reupload_every=data_reupload_every,
        weights_last_layer_data_re=legacy_last,
    )

    circuit = Circuit(
        n_qubits=3,
        n_layers=2,
        ansatz_name="basic_entangling",
        config=CircuitConfig(
            data_reupload_every=data_reupload_every,
            angle_scaling_method="scale_with_bias",
            angle_scaling=torch.pi,
            basis_angle_embedding="Y",
        ),
        weights=torq_weights,
        weights_last_layer_data_re=torq_last,
    )
    torq_out = circuit(x_torq)

    _assert_output_and_gradient_parity(
        legacy_out,
        torq_out,
        (x_legacy, legacy_weights, legacy_last),
        (x_torq, circuit.params, circuit.params_last_layer_reupload),
        atol=1e-5,
        rtol=1e-5,
    )
