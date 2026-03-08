import math

import pytest

np = pytest.importorskip("numpy")
qml = pytest.importorskip("pennylane")
torch = pytest.importorskip("torch")

import torq as tq
from torq.simple import Circuit, CircuitConfig


def _state_prep(state_vector, wires):
    if hasattr(qml, "StatePrep"):
        qml.StatePrep(state_vector, wires=wires)
    else:
        qml.QubitStateVector(state_vector, wires=wires)


def _single_pauli_op(pauli, qubit):
    if pauli == "I":
        return qml.Identity(wires=qubit)
    if pauli == "X":
        return qml.PauliX(wires=qubit)
    if pauli == "Y":
        return qml.PauliY(wires=qubit)
    if pauli == "Z":
        return qml.PauliZ(wires=qubit)
    raise ValueError(f"Unsupported Pauli operator {pauli!r}.")


def _qml_ops_from_pauli_string(pauli_spec: str, n_qubits: int):
    cleaned = pauli_spec.upper().replace(" ", "")
    groups = cleaned.split("_")
    qml_ops = []
    for group in groups:
        for start in range(n_qubits - len(group) + 1):
            op = _single_pauli_op(group[0], start)
            for offset, pauli in enumerate(group[1:], start=1):
                op = op @ _single_pauli_op(pauli, start + offset)
            qml_ops.append(op)
    return qml_ops


def _qml_ops_from_matrix_stack(matrices: torch.Tensor, n_qubits: int):
    wires = list(range(n_qubits))
    mats = matrices.detach().cpu().to(torch.complex128).numpy()
    return [qml.Hermitian(mat, wires=wires) for mat in mats]


def _pennylane_expectations_from_state_batch(
    states: torch.Tensor,
    qml_ops,
    *,
    unitary_layers: list[np.ndarray] | None = None,
) -> torch.Tensor:
    if states.dim() != 2:
        raise ValueError("states must have shape [B, 2**n].")
    n_qubits = int(math.log2(states.shape[1]))
    if 2 ** n_qubits != states.shape[1]:
        raise ValueError("State dimension must be a power of two.")

    device = qml.device("default.qubit", wires=n_qubits)
    all_wires = list(range(n_qubits))
    unitary_layers = [] if unitary_layers is None else unitary_layers

    @qml.qnode(device)
    def _circuit(state_vector):
        _state_prep(state_vector, wires=all_wires)
        for unitary in unitary_layers:
            qml.QubitUnitary(unitary, wires=all_wires)
        return [qml.expval(op) for op in qml_ops]

    states_np = states.detach().cpu().to(torch.complex128).numpy()
    out = []
    for state in states_np:
        out.append(np.asarray(_circuit(state), dtype=np.float64).reshape(-1))
    return torch.from_numpy(np.stack(out, axis=0)).to(torch.float64)


@pytest.mark.full
def test_measure_matches_pennylane_for_grouped_pauli_string():
    torch.manual_seed(7)
    batch_size = 3
    n_qubits = 3
    dim = 2 ** n_qubits

    raw = torch.complex(
        torch.randn(batch_size, dim, dtype=torch.float64),
        torch.randn(batch_size, dim, dtype=torch.float64),
    )
    states = raw / torch.linalg.norm(raw, dim=1, keepdim=True)

    observable = "z_zz_x"
    torq_out = tq.measure(states, observable, pauli_chunk_size=2).to(torch.float64)
    pennylane_out = _pennylane_expectations_from_state_batch(
        states,
        _qml_ops_from_pauli_string(observable, n_qubits),
    )

    assert torch.allclose(torq_out, pennylane_out, atol=1e-6, rtol=1e-6)


@pytest.mark.full
def test_measure_matches_pennylane_for_matrix_stack():
    torch.manual_seed(9)
    batch_size = 3
    n_qubits = 3
    dim = 2 ** n_qubits

    raw = torch.complex(
        torch.randn(batch_size, dim, dtype=torch.float64),
        torch.randn(batch_size, dim, dtype=torch.float64),
    )
    states = raw / torch.linalg.norm(raw, dim=1, keepdim=True)

    random_matrix = torch.complex(
        torch.randn(3, dim, dim, dtype=torch.float64),
        torch.randn(3, dim, dim, dtype=torch.float64),
    )
    matrices = random_matrix + random_matrix.conj().transpose(-2, -1)

    torq_out = tq.measure(states, matrices).to(torch.float64)
    pennylane_out = _pennylane_expectations_from_state_batch(
        states,
        _qml_ops_from_matrix_stack(matrices, n_qubits),
    )

    assert torch.allclose(torq_out, pennylane_out, atol=1e-6, rtol=1e-6)


@pytest.mark.full
def test_qlayer_pauli_string_matches_pennylane():
    torch.manual_seed(11)
    n_qubits = 3
    n_layers = 2
    batch_size = 2

    observable = "z_zz_x"
    config = CircuitConfig(
        observables=observable,
        pauli_measurement_chunk_size=2,
    )
    circuit = Circuit(
        n_qubits=n_qubits,
        n_layers=n_layers,
        ansatz_name="basic_entangling",
        config=config,
    )
    x = torch.rand(batch_size, n_qubits)
    torq_out = circuit(x).to(torch.float64)

    initial_states = tq.angle_embedding(
        x,
        angle_scaling_method=circuit.layer.angle_scaling_method,
        angle_scaling=circuit.layer.angle_scaling,
        basis=circuit.layer.basis_angle_embedding,
    ).squeeze(-1)

    layer_unitaries = []
    for layer_idx in range(n_layers):
        w = circuit.layer.params[layer_idx, 0]
        unitary = circuit.layer.ansatz.layer_op(layer_idx, w).detach().cpu().to(torch.complex128).numpy()
        layer_unitaries.append(unitary)

    pennylane_out = _pennylane_expectations_from_state_batch(
        initial_states,
        _qml_ops_from_pauli_string(observable, n_qubits),
        unitary_layers=layer_unitaries,
    )
    assert torch.allclose(torq_out, pennylane_out, atol=1e-6, rtol=1e-6)


@pytest.mark.full
def test_qlayer_matrix_stack_matches_pennylane():
    torch.manual_seed(13)
    n_qubits = 3
    n_layers = 2
    batch_size = 2
    dim = 2 ** n_qubits

    random_matrix = torch.complex(
        torch.randn(2, dim, dim, dtype=torch.float64),
        torch.randn(2, dim, dim, dtype=torch.float64),
    )
    matrices = random_matrix + random_matrix.conj().transpose(-2, -1)

    config = CircuitConfig(observables=matrices)
    circuit = Circuit(
        n_qubits=n_qubits,
        n_layers=n_layers,
        ansatz_name="basic_entangling",
        config=config,
    )
    x = torch.rand(batch_size, n_qubits)
    torq_out = circuit(x).to(torch.float64)

    initial_states = tq.angle_embedding(
        x,
        angle_scaling_method=circuit.layer.angle_scaling_method,
        angle_scaling=circuit.layer.angle_scaling,
        basis=circuit.layer.basis_angle_embedding,
    ).squeeze(-1)

    layer_unitaries = []
    for layer_idx in range(n_layers):
        w = circuit.layer.params[layer_idx, 0]
        unitary = circuit.layer.ansatz.layer_op(layer_idx, w).detach().cpu().to(torch.complex128).numpy()
        layer_unitaries.append(unitary)

    pennylane_out = _pennylane_expectations_from_state_batch(
        initial_states,
        _qml_ops_from_matrix_stack(matrices, n_qubits),
        unitary_layers=layer_unitaries,
    )
    assert torch.allclose(torq_out, pennylane_out, atol=1e-6, rtol=1e-6)
