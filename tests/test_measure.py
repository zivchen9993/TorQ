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
    with pytest.raises(ValueError, match="Matrix observable|local_observable|Matrix observables"):
        measure(state, torch.ones(2, 2, 2, 2))


@pytest.mark.full
def test_measure_accepts_case_insensitive_pauli_with_underscores():
    state = torch.zeros(1, 8, dtype=torch.complex64)
    state[:, 0] = 1.0  # |000>
    out = measure(state, "z_i_y")
    expected = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(out, expected, atol=1e-7, rtol=1e-7)


@pytest.mark.full
def test_measure_supports_grouped_pauli_words():
    state = torch.zeros(1, 16, dtype=torch.complex64)
    state[:, 0] = 1.0  # |0000>
    out = measure(state, "z_zz_x")
    expected = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(out, expected, atol=1e-7, rtol=1e-7)


@pytest.mark.full
def test_measure_rejects_all_identity_pauli_string():
    state = torch.zeros(1, 8, dtype=torch.complex64)
    with pytest.raises(ValueError, match="all identity"):
        measure(state, "i_i_i")


@pytest.mark.full
def test_measure_supports_global_matrix_and_matrices():
    state = torch.zeros(1, 4, dtype=torch.complex64)
    state[:, 0] = 1.0  # |00>
    sigma_z = sigma_Z_like(x=state)
    global_matrix = torch.kron(sigma_z, sigma_z)

    out_single = measure(state, global_matrix)
    assert torch.allclose(out_single, torch.tensor([[1.0]]), atol=1e-7, rtol=1e-7)

    stacked = torch.stack([global_matrix, global_matrix], dim=0)
    out_many = measure(state, stacked)
    assert torch.allclose(out_many, torch.tensor([[1.0, 1.0]]), atol=1e-7, rtol=1e-7)
