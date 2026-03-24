import importlib

import pytest

torch = pytest.importorskip("torch")

import torq as tq
from torq.Ansatz import make_ansatz as make_torq_ansatz
from torq.simple import Circuit, CircuitConfig
from _legacy_utils import (
    LEGACY_PACKAGE_NAME,
    legacy_basic_entangling_forward,
    legacy_scaling_kwargs,
    load_legacy_ansatz_module,
    load_legacy_quantum_lib,
)


@pytest.fixture(scope="module")
def legacy_qg():
    return load_legacy_quantum_lib()


@pytest.fixture(scope="module")
def legacy_ansatz_module(legacy_qg):
    return load_legacy_ansatz_module()


@pytest.mark.full
@pytest.mark.parametrize("angle_scaling_method", ["none", "scale", "scale_with_bias", "asin", "acos"])
@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_angle_embedding_matches_legacy(legacy_qg, angle_scaling_method, basis):
    torch.manual_seed(17)
    x = torch.rand(4, 3, dtype=torch.float64) * 1.8 - 0.9
    legacy_kwargs = legacy_scaling_kwargs(angle_scaling_method, torch.pi)

    legacy_state = legacy_qg.angle_embedding(x, basis=basis, **legacy_kwargs)
    torq_state = tq.angle_embedding(
        x,
        angle_scaling_method=angle_scaling_method,
        angle_scaling=torch.pi,
        basis=basis,
    )

    assert torch.allclose(legacy_state, torq_state, atol=1e-6, rtol=1e-6)


@pytest.mark.full
@pytest.mark.parametrize("angle_scaling_method", ["none", "scale", "scale_with_bias", "asin", "acos"])
def test_data_reuploading_matches_legacy(legacy_qg, angle_scaling_method):
    torch.manual_seed(23)
    x = torch.rand(3, 4, dtype=torch.float64) * 1.8 - 0.9
    legacy_kwargs = legacy_scaling_kwargs(angle_scaling_method, torch.pi)

    legacy_gates, legacy_batch_size, legacy_n_qubits = legacy_qg.data_reuploading_gates(
        x,
        basis="Y",
        **legacy_kwargs,
    )
    torq_gates = tq.data_reuploading_gates(
        x,
        angle_scaling_method=angle_scaling_method,
        angle_scaling=torch.pi,
        basis="Y",
    )

    state = torch.complex(
        torch.randn(legacy_batch_size, 2 ** legacy_n_qubits, dtype=torch.float64),
        torch.randn(legacy_batch_size, 2 ** legacy_n_qubits, dtype=torch.float64),
    )
    state = state / torch.linalg.norm(state, dim=1, keepdim=True)

    legacy_out = legacy_qg.data_reuploading(
        legacy_batch_size,
        legacy_n_qubits,
        state.clone(),
        legacy_gates,
    )
    torq_out = tq.data_reuploading(state.clone(), torq_gates)

    assert torch.allclose(legacy_gates, torq_gates, atol=1e-6, rtol=1e-6)
    assert torch.allclose(legacy_out, torq_out, atol=1e-6, rtol=1e-6)


@pytest.mark.full
def test_measurement_matches_legacy_local_z(legacy_qg):
    torch.manual_seed(31)
    raw = torch.complex(
        torch.randn(5, 8, dtype=torch.float64),
        torch.randn(5, 8, dtype=torch.float64),
    )
    state = raw / torch.linalg.norm(raw, dim=1, keepdim=True)
    observable = legacy_qg.sigma_Z_like(x=state)

    legacy_out = legacy_qg.measure(state, observable)

    assert torch.allclose(legacy_out, tq.measure_local_Z(state), atol=1e-6, rtol=1e-6)
    assert torch.allclose(legacy_out, tq.measure(state), atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        legacy_qg.measure_local(state, observable),
        tq.measure_local_observable(state, observable),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.full
def test_basic_entangling_matches_legacy_strongly_entangling(legacy_qg):
    torch.manual_seed(37)
    weights = torch.randn(4, 3, dtype=torch.float32)

    legacy_ansatz = importlib.import_module(f"{LEGACY_PACKAGE_NAME}.Ansatz").make_ansatz("strongly_entangling", 4, 3)
    torq_ansatz = make_torq_ansatz("basic_entangling", 4, 3)

    legacy_layer = legacy_ansatz.layer_op(2, weights)
    torq_layer = torq_ansatz.layer_op(2, weights)

    assert torch.allclose(legacy_layer, torq_layer, atol=1e-6, rtol=1e-6)


@pytest.mark.full
def test_strongly_entangling_matches_legacy_all_to_all_variant(legacy_qg):
    torch.manual_seed(41)
    weights = torch.randn(4, 3, dtype=torch.float32)

    legacy_ansatz = importlib.import_module(f"{LEGACY_PACKAGE_NAME}.Ansatz").StronglyEntanglingAllToAll(4, 3)
    torq_ansatz = make_torq_ansatz("strongly_entangling", 4, 3)

    legacy_layer = legacy_ansatz.layer_op(2, weights)
    torq_layer = torq_ansatz.layer_op(2, weights)

    assert torch.allclose(legacy_layer, torq_layer, atol=1e-6, rtol=1e-6)


@pytest.mark.full
@pytest.mark.parametrize("ansatz_name", ["cross_mesh", "cross_mesh_2_rots", "cross_mesh_cx_rot", "no_entanglement_ansatz"])
def test_shared_ansatz_variants_match_legacy(legacy_ansatz_module, ansatz_name):
    torch.manual_seed(43)
    n_qubits = 3
    n_layers = 2

    if ansatz_name == "cross_mesh":
        weights = torch.randn(n_qubits ** 2, dtype=torch.float32)
    elif ansatz_name == "cross_mesh_2_rots":
        weights = torch.randn(n_qubits + n_qubits ** 2, dtype=torch.float32)
    else:
        weights = torch.randn(n_qubits, 3, dtype=torch.float32)

    legacy_ansatz = legacy_ansatz_module.make_ansatz(ansatz_name, n_qubits, n_layers)
    torq_ansatz = make_torq_ansatz(ansatz_name, n_qubits, n_layers)

    legacy_layer = legacy_ansatz.layer_op(1, weights)
    torq_layer = torq_ansatz.layer_op(1, weights)

    assert torch.allclose(legacy_layer, torq_layer, atol=1e-6, rtol=1e-6)


@pytest.mark.full
def test_basic_entangling_circuit_matches_legacy_manual_forward(legacy_qg, legacy_ansatz_module):
    torch.manual_seed(47)
    weights = torch.randn(2, 1, 3, 3, dtype=torch.float32)
    x = torch.rand(4, 3, dtype=torch.float32) * 1.8 - 0.9

    config = CircuitConfig(
        angle_scaling_method="scale",
        angle_scaling=torch.pi,
        basis_angle_embedding="X",
    )
    circuit = Circuit(
        n_qubits=3,
        n_layers=2,
        ansatz_name="basic_entangling",
        config=config,
        weights=weights,
    )

    torq_out = circuit(x)
    legacy_out = legacy_basic_entangling_forward(
        legacy_qg,
        legacy_ansatz_module,
        x,
        weights,
        angle_scaling_method="scale",
        angle_scaling=torch.pi,
        basis="X",
    )

    assert torch.allclose(legacy_out, torq_out, atol=1e-6, rtol=1e-6)


@pytest.mark.full
def test_basic_entangling_data_reupload_circuit_matches_legacy_manual_forward(legacy_qg, legacy_ansatz_module):
    torch.manual_seed(53)
    data_reupload_every = 2
    weights = torch.randn(2, data_reupload_every, 3, 3, dtype=torch.float32)
    weights_last_layer_data_re = torch.randn(data_reupload_every, 3, 3, dtype=torch.float32)
    x = torch.rand(4, 3, dtype=torch.float32) * 1.8 - 0.9

    config = CircuitConfig(
        data_reupload_every=data_reupload_every,
        angle_scaling_method="scale_with_bias",
        angle_scaling=torch.pi,
        basis_angle_embedding="Y",
    )
    circuit = Circuit(
        n_qubits=3,
        n_layers=2,
        ansatz_name="basic_entangling",
        config=config,
        weights=weights,
        weights_last_layer_data_re=weights_last_layer_data_re,
    )

    torq_out = circuit(x)
    legacy_out = legacy_basic_entangling_forward(
        legacy_qg,
        legacy_ansatz_module,
        x,
        weights,
        angle_scaling_method="scale_with_bias",
        angle_scaling=torch.pi,
        basis="Y",
        data_reupload_every=data_reupload_every,
        weights_last_layer_data_re=weights_last_layer_data_re,
    )

    assert torch.allclose(legacy_out, torq_out, atol=1e-6, rtol=1e-6)
