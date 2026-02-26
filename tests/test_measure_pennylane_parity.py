import math

import pytest

np = pytest.importorskip("numpy")
qml = pytest.importorskip("pennylane")
torch = pytest.importorskip("torch")

import torq as tq
from torq.Measure import measure_observables
from torq.simple import Circuit, CircuitConfig


def _state_prep(state_vector, wires):
    if hasattr(qml, "StatePrep"):
        qml.StatePrep(state_vector, wires=wires)
    else:
        qml.QubitStateVector(state_vector, wires=wires)


def _normalize_wires(wires):
    if isinstance(wires, int):
        return [wires]
    return list(wires)


def _normalize_pauli_ops(pauli, n_wires):
    if isinstance(pauli, str):
        cleaned = pauli.replace(" ", "").upper()
        if len(cleaned) == 1:
            return [cleaned] * n_wires
        if len(cleaned) == n_wires:
            return list(cleaned)
        raise ValueError(f"Invalid pauli string length {len(cleaned)} for n_wires={n_wires}.")
    ops = [str(op).upper() for op in pauli]
    if len(ops) != n_wires:
        raise ValueError(f"Pauli sequence length {len(ops)} must equal n_wires={n_wires}.")
    return ops


def _single_pauli_op(pauli, wire):
    if pauli == "I":
        return qml.Identity(wires=wire)
    if pauli == "X":
        return qml.PauliX(wires=wire)
    if pauli == "Y":
        return qml.PauliY(wires=wire)
    if pauli == "Z":
        return qml.PauliZ(wires=wire)
    raise ValueError(f"Unsupported Pauli operator {pauli!r}.")


def _qml_operator_and_coeff(spec):
    wires = _normalize_wires(spec["wires"])
    has_pauli = spec.get("pauli") is not None
    has_matrix = spec.get("matrix") is not None
    if has_pauli == has_matrix:
        raise ValueError("Each observable must set exactly one of 'pauli' or 'matrix'.")

    coeff = float(spec.get("coeff", 1.0))
    if has_pauli:
        pauli_ops = _normalize_pauli_ops(spec["pauli"], len(wires))
        ops = [_single_pauli_op(pauli, wire) for pauli, wire in zip(pauli_ops, wires)]
        op = ops[0]
        for next_op in ops[1:]:
            op = op @ next_op
        return op, coeff

    matrix = np.asarray(spec["matrix"], dtype=np.complex128)
    return qml.Hermitian(matrix, wires=wires), coeff


def _pennylane_expectations_from_state_batch(
    states: torch.Tensor,
    observables,
    *,
    unitary_layers: list[np.ndarray] | None = None,
) -> torch.Tensor:
    if states.dim() != 2:
        raise ValueError("states must have shape [B, 2**n].")
    n_qubits = int(math.log2(states.shape[1]))
    if 2 ** n_qubits != states.shape[1]:
        raise ValueError("State dimension must be a power of two.")

    qml_ops = []
    coeffs = []
    for spec in observables:
        op, coeff = _qml_operator_and_coeff(spec)
        qml_ops.append(op)
        coeffs.append(coeff)
    coeffs = np.asarray(coeffs, dtype=np.float64)
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
        vals = np.asarray(_circuit(state), dtype=np.float64).reshape(-1) * coeffs
        out.append(vals)
    return torch.from_numpy(np.stack(out, axis=0)).to(torch.float64)


@pytest.mark.full
def test_measure_observables_matches_pennylane_for_mixed_observables():
    torch.manual_seed(7)
    batch_size = 3
    n_qubits = 3
    dim = 2 ** n_qubits

    raw = torch.complex(
        torch.randn(batch_size, dim, dtype=torch.float64),
        torch.randn(batch_size, dim, dtype=torch.float64),
    )
    states = raw / torch.linalg.norm(raw, dim=1, keepdim=True)

    random_matrix = torch.complex(
        torch.randn(4, 4, dtype=torch.float64),
        torch.randn(4, 4, dtype=torch.float64),
    )
    hermitian = random_matrix + random_matrix.conj().mT

    observables = [
        {"wires": [0], "pauli": "Z"},
        {"wires": [0, 1], "pauli": "ZZ"},
        {"wires": [2], "pauli": "X"},
        {"wires": [0, 2], "pauli": ["Y", "X"], "coeff": 0.5},
        {"wires": [1, 2], "matrix": hermitian, "coeff": 0.25},
    ]

    torq_out = measure_observables(states, observables, pauli_chunk_size=2).to(torch.float64)
    pennylane_out = _pennylane_expectations_from_state_batch(states, observables)

    assert torch.allclose(torq_out, pennylane_out, atol=1e-6, rtol=1e-6)


@pytest.mark.full
def test_qlayer_measurement_observables_match_pennylane():
    torch.manual_seed(11)
    n_qubits = 3
    n_layers = 2
    batch_size = 2

    observables = [
        {"wires": [0], "pauli": "Z"},
        {"wires": [1], "pauli": "X"},
        {"wires": [0, 1], "pauli": "ZZ"},
        {"wires": [0, 2], "pauli": ["Y", "X"], "coeff": 0.75},
    ]
    config = CircuitConfig(
        measurement_observables=observables,
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
        observables,
        unitary_layers=layer_unitaries,
    )
    assert torch.allclose(torq_out, pennylane_out, atol=1e-6, rtol=1e-6)
