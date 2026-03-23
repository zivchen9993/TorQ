import pytest

torch = pytest.importorskip("torch")

import torq as tq
from torq.Ansatz import make_ansatz
from torq.simple import Circuit, CircuitConfig


def _manual_cnot_layer(n_qubits: int, pairs: list[tuple[int, int]], weights: torch.Tensor) -> torch.Tensor:
    gates = [
        tq.get_cnot_ops(n_qubits, control_qubit_id, target_qubit_id, x=weights).unsqueeze(0)
        for control_qubit_id, target_qubit_id in pairs
    ]
    return tq.multi_dim_matmul_reversed(*gates)


@pytest.mark.full
def test_single_rot_basic_ent_weight_shape_supported_without_data_reupload():
    weights = torch.rand(2, 3)
    circuit = Circuit(
        n_qubits=3,
        n_layers=2,
        ansatz_name="single_rot_basic_ent",
        config=CircuitConfig(data_reupload_every=0),
        weights=weights,
    )
    assert tuple(circuit.params.shape) == (2, 1, 3)


@pytest.mark.full
def test_tile_weight_shapes_supported_for_both_rotation_modes():
    circuit_three_param = Circuit(
        n_qubits=3,
        n_layers=2,
        ansatz_name="tile",
        config=CircuitConfig(tile_rotation_params=3),
        weights=torch.rand(2, 3, 3),
    )
    circuit_single_param = Circuit(
        n_qubits=3,
        n_layers=2,
        ansatz_name="tile",
        config=CircuitConfig(tile_rotation_params=1),
        weights=torch.rand(2, 3),
    )

    assert tuple(circuit_three_param.params.shape) == (2, 1, 3, 3)
    assert tuple(circuit_single_param.params.shape) == (2, 1, 3)


@pytest.mark.full
def test_single_rot_basic_ent_rotation_axis_matches_manual_layer():
    weights = torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64)
    config = CircuitConfig(single_rotation_gate="ry")

    ansatz = make_ansatz("single_rot_basic_ent", 3, 2, config=config)
    actual = ansatz.layer_op(0, weights)
    expected = tq.basic_or_strongly_single_layer(
        3,
        weights,
        tq.get_cnot_ladder(3, r=0, x=weights),
        sigma_single_rot=tq.get_ry,
    )

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)


@pytest.mark.full
def test_tile_noncyclic_layer_matches_manual_construction():
    weights = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float64)
    config = CircuitConfig(
        single_rotation_gate="rx",
        tile_rotation_params=1,
        tile_sublayers=2,
        tile_cyclic=False,
    )

    ansatz = make_ansatz("tile", 4, 1, config=config)
    actual = ansatz.layer_op(0, weights)
    expected = tq.basic_or_strongly_single_layer(
        4,
        weights,
        _manual_cnot_layer(4, [(0, 1), (2, 3), (1, 2), (0, 1), (2, 3), (1, 2)], weights),
        sigma_single_rot=tq.get_rx,
    )

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)


@pytest.mark.full
def test_tile_cyclic_layer_matches_manual_construction():
    weights = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5], dtype=torch.float64)
    config = CircuitConfig(
        single_rotation_gate="rz",
        tile_rotation_params=1,
        tile_sublayers=1,
        tile_cyclic=True,
    )

    ansatz = make_ansatz("tile", 5, 1, config=config)
    actual = ansatz.layer_op(0, weights)
    expected = tq.basic_or_strongly_single_layer(
        5,
        weights,
        _manual_cnot_layer(5, [(0, 1), (2, 3), (1, 2), (3, 4), (4, 0)], weights),
        sigma_single_rot=tq.get_rz,
    )

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)


@pytest.mark.full
def test_circuit_config_validates_new_ansatz_options():
    with pytest.raises(ValueError, match="single_rotation_gate"):
        CircuitConfig(single_rotation_gate="sx")
    with pytest.raises(ValueError, match="tile_rotation_params"):
        CircuitConfig(tile_rotation_params=2)
    with pytest.raises(ValueError, match="tile_sublayers"):
        CircuitConfig(tile_sublayers=0)
