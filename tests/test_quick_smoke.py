import pytest

torch = pytest.importorskip("torch")

from torq.simple import Circuit, CircuitConfig


@pytest.mark.quick
@pytest.mark.parametrize(
    "ansatz_name",
    [
        "basic_entangling",
        "strongly_entangling",
        "cross_mesh",
        "cross_mesh_2_rots",
        "cross_mesh_cx_rot",
        "no_entanglement_ansatz",
    ],
)
def test_all_ansatzes_forward_smoke(ansatz_name):
    circuit = Circuit(
        n_qubits=3,
        n_layers=2,
        ansatz_name=ansatz_name,
        config=CircuitConfig(data_reupload_every=0),
    )
    x = torch.rand(4, 3)
    y = circuit(x)
    assert y.shape == (4, 3)
    assert torch.isfinite(y).all()


@pytest.mark.quick
def test_data_reupload_smoke_basic_entangling():
    circuit = Circuit(
        n_qubits=3,
        n_layers=2,
        ansatz_name="basic_entangling",
        config=CircuitConfig(data_reupload_every=2),
    )
    x = torch.rand(2, 3)
    y = circuit(x)
    assert y.shape == (2, 3)
    assert circuit.params_last_layer_reupload is not None
    assert torch.isfinite(y).all()
