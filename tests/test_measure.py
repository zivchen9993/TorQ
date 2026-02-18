import pytest

torch = pytest.importorskip("torch")

from torq.Measure import measure
from torq.SingleQubitGates import sigma_X_like, sigma_Z_like


@pytest.mark.quick
def test_measure_defaults_to_pauli_z():
    state = torch.zeros(1, 4, dtype=torch.complex64)
    state[:, 0] = 1.0  # |00>
    out = measure(state)
    expected = torch.tensor([[1.0, 1.0]])
    assert torch.allclose(out, expected)


@pytest.mark.full
def test_measure_supports_shared_local_observable():
    state = torch.zeros(1, 4, dtype=torch.complex64)
    state[:, 0] = 1.0  # |00>
    sigma_x = sigma_X_like(x=state)
    out = measure(state, sigma_x)
    expected = torch.zeros_like(out)
    assert torch.allclose(out, expected, atol=1e-7, rtol=1e-7)


@pytest.mark.full
def test_measure_supports_per_qubit_local_observable():
    state = torch.zeros(1, 4, dtype=torch.complex64)
    state[:, 0] = 1.0  # |00>
    sigma_x = sigma_X_like(x=state)
    sigma_z = sigma_Z_like(x=state)
    observable = torch.stack([sigma_x, sigma_z], dim=0)
    out = measure(state, observable)
    expected = torch.tensor([[0.0, 1.0]])
    assert torch.allclose(out, expected, atol=1e-7, rtol=1e-7)


@pytest.mark.full
def test_measure_rejects_bad_observable_shape():
    state = torch.zeros(1, 4, dtype=torch.complex64)
    state[:, 0] = 1.0
    with pytest.raises(ValueError, match="local_observable"):
        measure(state, torch.ones(2, 2, 2, 2))
