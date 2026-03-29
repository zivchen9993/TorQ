import pytest
import warnings

torch = pytest.importorskip("torch")

from torq._pennylane_backend import (
    _PennyLaneRunner,
    _normalize_pennylane_output,
)


def test_normalize_pennylane_output_transposes_measurement_major_tensor():
    raw = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    out = _normalize_pennylane_output(raw, batch_size=3, n_outputs=2)

    assert out.shape == (3, 2)
    assert torch.equal(out, raw.transpose(0, 1))


def test_normalize_pennylane_output_preserves_batch_major_tensor():
    raw = torch.tensor(
        [
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
        ]
    )

    out = _normalize_pennylane_output(raw, batch_size=3, n_outputs=2)

    assert out.shape == (3, 2)
    assert torch.equal(out, raw)


def test_normalize_pennylane_output_stacks_sequence_measurements():
    raw = (
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([4.0, 5.0, 6.0]),
    )

    out = _normalize_pennylane_output(raw, batch_size=3, n_outputs=2)

    assert out.shape == (3, 2)
    assert torch.equal(out, torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))


def test_normalize_pennylane_output_rejects_unknown_shape():
    raw = torch.ones(4, 5)

    with pytest.raises(ValueError, match="cannot be normalized"):
        _normalize_pennylane_output(raw, batch_size=3, n_outputs=2)


def test_runner_measures_complex_state_outputs():
    captured = {}

    def circuit(_x):
        return torch.tensor(
            [
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ],
            dtype=torch.complex64,
        )

    def measure_state(state):
        captured["shape"] = tuple(state.shape)
        return torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)

    runner = _PennyLaneRunner(
        circuit=circuit,
        angle_scaling_method="none",
        angle_scaling=1.0,
        basis_angle_embedding="X",
        n_outputs=2,
        measure_state=measure_state,
    )

    out = runner.forward(torch.rand(2, 2))

    assert captured["shape"] == (2, 4)
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))


def test_runner_uses_measurement_circuit_when_available():
    captured = {}

    def measurement_circuit(x):
        captured["scaled_shape"] = tuple(x.shape)
        return (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))

    def measurement_formatter(result):
        captured["formatter_called"] = True
        return torch.stack(result, dim=-1)

    runner = _PennyLaneRunner(
        circuit=None,
        angle_scaling_method="none",
        angle_scaling=1.0,
        basis_angle_embedding="X",
        n_outputs=2,
        measurement_circuit=measurement_circuit,
        measurement_result_formatter=measurement_formatter,
    )

    out = runner.forward(torch.rand(2, 2))

    assert captured == {"scaled_shape": (2, 2), "formatter_called": True}
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.tensor([[1.0, 3.0], [2.0, 4.0]]))


def test_pennylane_backend_smoke_with_state_returning_torq_bench():
    pytest.importorskip("pennylane")
    pytest.importorskip("torq_bench")

    from torq.simple import Circuit, CircuitConfig

    cfg = CircuitConfig(
        pennylane_backend=True,
        pennylane_dev_name="default.qubit",
    )
    model = Circuit(
        n_qubits=2,
        n_layers=1,
        ansatz_name="strongly_entangling",
        config=cfg,
    )

    out = model(torch.rand(2, 2))

    assert out.shape == (2, 2)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_pennylane_backend_supports_non_x_basis_and_global_observables():
    pytest.importorskip("pennylane")
    pytest.importorskip("torq_bench")

    from torq.simple import Circuit, CircuitConfig

    cfg = CircuitConfig(
        pennylane_backend=True,
        pennylane_dev_name="default.qubit",
        basis_angle_embedding="Y",
        observables="XX",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = Circuit(
            n_qubits=2,
            n_layers=1,
            ansatz_name="strongly_entangling",
            config=cfg,
        )
        out = model(torch.rand(3, 2))

    fallback_warnings = [w for w in caught if "using TorQ backend" in str(w.message)]
    assert not fallback_warnings
    assert out.shape == (3, 1)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    ("ansatz_name", "n_qubits", "config_kwargs"),
    [
        ("single_rot_basic_ent", 2, {"single_rotation_gate": "ry"}),
        (
            "tile",
            4,
            {
                "tile_rotation_params": 1,
                "single_rotation_gate": "rz",
                "tile_sublayers": 2,
                "tile_cyclic": True,
            },
        ),
        ("no_entanglement_ansatz", 2, {}),
    ],
)
def test_pennylane_backend_supports_extended_ansatzes(
    ansatz_name,
    n_qubits,
    config_kwargs,
):
    pytest.importorskip("pennylane")
    pytest.importorskip("torq_bench")

    from torq.simple import Circuit, CircuitConfig

    cfg = CircuitConfig(
        pennylane_backend=True,
        pennylane_dev_name="default.qubit",
        **config_kwargs,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = Circuit(
            n_qubits=n_qubits,
            n_layers=1,
            ansatz_name=ansatz_name,
            config=cfg,
        )
        out = model(torch.rand(2, n_qubits))

    fallback_warnings = [w for w in caught if "using TorQ backend" in str(w.message)]
    assert not fallback_warnings
    assert out.shape == (2, n_qubits)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
