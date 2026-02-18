import pytest

torch = pytest.importorskip("torch")

from torq.simple import Circuit, CircuitConfig


@pytest.mark.full
def test_legacy_angle_aliases_map_to_scaling_method():
    assert CircuitConfig(angle_asin=True).angle_scaling_method == "asin"
    assert CircuitConfig(angle_acos=True).angle_scaling_method == "acos"
    assert CircuitConfig(scale_with_bias=True).angle_scaling_method == "scale_with_bias"


@pytest.mark.full
def test_legacy_angle_aliases_are_mutually_exclusive():
    with pytest.raises(ValueError, match="Only one of angle_asin/angle_acos/scale_with_bias"):
        CircuitConfig(angle_asin=True, angle_acos=True)


@pytest.mark.full
def test_explicit_and_legacy_scaling_settings_cannot_be_mixed():
    with pytest.raises(ValueError, match="Use either angle_scaling_method or legacy flags"):
        CircuitConfig(angle_scaling_method="scale", angle_asin=True)


@pytest.mark.full
@pytest.mark.parametrize("angle_scaling_method", ["none", "scale", "scale_with_bias", "asin", "acos"])
@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_angle_scaling_methods_and_bases(angle_scaling_method, basis):
    cfg = CircuitConfig(
        angle_scaling_method=angle_scaling_method,
        angle_scaling=torch.pi,
    )
    circuit = Circuit(
        n_qubits=2,
        n_layers=1,
        ansatz_name="basic_entangling",
        config=cfg,
        basis_angle_embedding=basis,
    )
    x = torch.rand(3, 2) * 2 - 1
    y = circuit(x)
    assert y.shape == (3, 2)
    assert torch.isfinite(y).all()


@pytest.mark.full
@pytest.mark.parametrize(
    ("flag_name", "expected_value"),
    [
        ("init_identity", 0.0),
        ("init_ones", torch.pi),
        ("init_pi_half", torch.pi / 2),
    ],
)
def test_init_flags(flag_name, expected_value):
    cfg_kwargs = {flag_name: True}
    circuit = Circuit(
        n_qubits=2,
        n_layers=2,
        ansatz_name="basic_entangling",
        config=CircuitConfig(**cfg_kwargs),
    )
    expected = torch.full_like(circuit.params, expected_value)
    assert torch.allclose(circuit.params.detach(), expected)


@pytest.mark.full
def test_legacy_weight_shape_supported_without_data_reupload():
    weights = torch.rand(2, 3, 3)
    circuit = Circuit(
        n_qubits=3,
        n_layers=2,
        ansatz_name="basic_entangling",
        config=CircuitConfig(data_reupload_every=0),
        weights=weights,
    )
    assert tuple(circuit.params.shape) == (2, 1, 3, 3)


@pytest.mark.full
def test_requires_last_layer_weights_for_data_reupload():
    weights = torch.rand(2, 2, 3, 3)
    with pytest.raises(ValueError, match="weights_last_layer_data_re"):
        Circuit(
            n_qubits=3,
            n_layers=2,
            ansatz_name="basic_entangling",
            config=CircuitConfig(data_reupload_every=2),
            weights=weights,
        )


@pytest.mark.full
def test_weight_shape_validation_for_data_reupload():
    weights = torch.rand(2, 3, 3)
    weights_last = torch.rand(2, 3, 3)
    with pytest.raises(ValueError, match="weights has shape"):
        Circuit(
            n_qubits=3,
            n_layers=2,
            ansatz_name="basic_entangling",
            config=CircuitConfig(data_reupload_every=2),
            weights=weights,
            weights_last_layer_data_re=weights_last,
        )


@pytest.mark.full
def test_valid_weight_shapes_for_data_reupload():
    weights = torch.rand(2, 2, 3, 3)
    weights_last = torch.rand(2, 3, 3)
    circuit = Circuit(
        n_qubits=3,
        n_layers=2,
        ansatz_name="basic_entangling",
        config=CircuitConfig(data_reupload_every=2),
        weights=weights,
        weights_last_layer_data_re=weights_last,
    )
    y = circuit(torch.rand(2, 3))
    assert y.shape == (2, 3)
    assert torch.isfinite(y).all()
