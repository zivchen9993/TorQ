import pytest

torch = pytest.importorskip("torch")

import torq as tq
from torq.QLayer import QLayer
from torq.simple import Circuit, CircuitConfig


def _dense_forward(layer: QLayer, x: torch.Tensor) -> torch.Tensor:
    if not layer.data_reupload_every:
        state = tq.angle_embedding(
            x,
            angle_scaling_method=layer.angle_scaling_method,
            angle_scaling=layer.angle_scaling,
            basis=layer.basis_angle_embedding,
        ).squeeze(-1)
        data_gates = None
    else:
        state = tq.get_initial_state(
            n_qubits=layer.n_qubits,
            batch_size=x.shape[0],
            x=layer.params,
        ).squeeze(-1)
        data_gates = tq.data_reuploading_gates(
            x,
            angle_scaling_method=layer.angle_scaling_method,
            angle_scaling=layer.angle_scaling,
            basis=layer.basis_angle_embedding,
        )

    angles_reparametrize = None
    if layer.reparametrize_sin_cos:
        angles_reparametrize = torch.atan2(torch.sin(layer.params), torch.cos(layer.params))

    reps = max(layer.data_reupload_every, 1)
    for layer_idx in range(layer.n_layers):
        for d_reup in range(reps):
            if layer.reparametrize_sin_cos:
                w = angles_reparametrize[layer_idx, d_reup]
            else:
                w = layer.params[layer_idx, d_reup]
            idx = d_reup if layer.data_reupload_every else layer_idx
            state = tq.apply_matrix(state, layer.ansatz.layer_op(idx, w))

        if layer.data_reupload_every:
            state = tq.data_reuploading(state, data_gates).squeeze(-1)

    if layer.data_reupload_every:
        angles_reparametrize_last = None
        if layer.reparametrize_sin_cos:
            angles_reparametrize_last = torch.atan2(
                torch.sin(layer.params_last_layer_reupload),
                torch.cos(layer.params_last_layer_reupload),
            )
        for d_reup in range(layer.data_reupload_every):
            if layer.reparametrize_sin_cos:
                w = angles_reparametrize_last[d_reup]
            else:
                w = layer.params_last_layer_reupload[d_reup]
            state = tq.apply_matrix(state, layer.ansatz.layer_op(d_reup, w))

    return tq.measure(
        state,
        layer.observables,
        pauli_chunk_size=getattr(layer.config, "pauli_measurement_chunk_size", 8),
    )


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


@pytest.mark.quick
@pytest.mark.parametrize(
    ("ansatz_name", "data_reupload_every"),
    [
        ("basic_entangling", 0),
        ("basic_entangling", 2),
        ("strongly_entangling", 0),
        ("strongly_entangling", 2),
        ("no_entanglement_ansatz", 0),
    ],
)
def test_direct_execution_matches_dense_path(ansatz_name, data_reupload_every):
    torch.manual_seed(0)
    circuit = Circuit(
        n_qubits=4,
        n_layers=2,
        ansatz_name=ansatz_name,
        config=CircuitConfig(
            data_reupload_every=data_reupload_every,
            observables="z_zz_x",
            pauli_measurement_chunk_size=2,
        ),
    )
    x = torch.rand(3, 4)

    direct = circuit(x)
    dense = _dense_forward(circuit.layer, x)

    assert direct.shape == dense.shape
    assert torch.allclose(direct, dense, atol=1e-6, rtol=1e-6)
