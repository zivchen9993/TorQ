import importlib
import importlib.util
from pathlib import Path
import sys

import torch


LEGACY_PACKAGE_NAME = "quantum_lib"


def load_legacy_quantum_lib():
    root = Path(__file__).resolve().parents[1] / "quantum_lib_old"
    init_file = root / "__init__.py"

    existing = sys.modules.get(LEGACY_PACKAGE_NAME)
    if existing is not None and Path(getattr(existing, "__file__", "")) == init_file:
        return existing

    for name in list(sys.modules):
        if name == LEGACY_PACKAGE_NAME or name.startswith(f"{LEGACY_PACKAGE_NAME}."):
            del sys.modules[name]

    spec = importlib.util.spec_from_file_location(
        LEGACY_PACKAGE_NAME,
        init_file,
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load legacy package from {init_file}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[LEGACY_PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


def load_legacy_ansatz_module():
    load_legacy_quantum_lib()
    return importlib.import_module(f"{LEGACY_PACKAGE_NAME}.Ansatz")


def legacy_scaling_kwargs(angle_scaling_method: str, angle_scaling: float):
    if angle_scaling_method == "none":
        return {
            "scale": None,
            "asin": False,
            "acos": False,
            "scale_with_bias": False,
        }
    if angle_scaling_method == "scale":
        return {
            "scale": angle_scaling,
            "asin": False,
            "acos": False,
            "scale_with_bias": False,
        }
    if angle_scaling_method == "scale_with_bias":
        return {
            "scale": angle_scaling,
            "asin": False,
            "acos": False,
            "scale_with_bias": True,
        }
    if angle_scaling_method == "asin":
        return {
            "scale": None,
            "asin": True,
            "acos": False,
            "scale_with_bias": False,
        }
    if angle_scaling_method == "acos":
        return {
            "scale": None,
            "asin": False,
            "acos": True,
            "scale_with_bias": False,
        }
    raise ValueError(f"Unsupported angle_scaling_method {angle_scaling_method!r}.")


def legacy_basic_entangling_forward(
    legacy_qg,
    legacy_ansatz_module,
    x: torch.Tensor,
    weights: torch.Tensor,
    *,
    angle_scaling_method: str,
    angle_scaling: float,
    basis: str,
    data_reupload_every: int = 0,
    weights_last_layer_data_re: torch.Tensor | None = None,
) -> torch.Tensor:
    legacy_kwargs = legacy_scaling_kwargs(angle_scaling_method, angle_scaling)
    batch_size, n_qubits = x.shape
    n_layers = weights.shape[0]
    ansatz = legacy_ansatz_module.make_ansatz("strongly_entangling", n_qubits, n_layers)

    if data_reupload_every:
        state = legacy_qg.get_initial_state(
            n_qubits=n_qubits,
            batch_size=batch_size,
            x=weights,
        ).squeeze(-1)
        data_gates, legacy_batch_size, legacy_n_qubits = legacy_qg.data_reuploading_gates(
            x,
            basis=basis,
            **legacy_kwargs,
        )
    else:
        state = legacy_qg.angle_embedding(
            x,
            basis=basis,
            **legacy_kwargs,
        ).squeeze(-1)
        data_gates = None
        legacy_batch_size = batch_size
        legacy_n_qubits = n_qubits

    reps = max(data_reupload_every, 1)
    for layer_idx in range(n_layers):
        for reupload_idx in range(reps):
            w = weights[layer_idx, reupload_idx]
            legacy_layer_idx = reupload_idx if data_reupload_every else layer_idx
            state = legacy_qg.apply_matrix(state, ansatz.layer_op(legacy_layer_idx, w))

        if data_reupload_every:
            state = legacy_qg.data_reuploading(
                legacy_batch_size,
                legacy_n_qubits,
                state,
                data_gates,
            ).squeeze(-1)

    if data_reupload_every:
        assert weights_last_layer_data_re is not None
        for reupload_idx in range(data_reupload_every):
            state = legacy_qg.apply_matrix(
                state,
                ansatz.layer_op(reupload_idx, weights_last_layer_data_re[reupload_idx]),
            )

    return legacy_qg.measure(state, legacy_qg.sigma_Z_like(x=state))


def legacy_strongly_entangling_all_to_all_forward(
    legacy_qg,
    legacy_ansatz_module,
    x: torch.Tensor,
    weights: torch.Tensor,
    *,
    angle_scaling_method: str,
    angle_scaling: float,
    basis: str,
) -> torch.Tensor:
    legacy_kwargs = legacy_scaling_kwargs(angle_scaling_method, angle_scaling)
    n_layers = weights.shape[0]
    ansatz = legacy_ansatz_module.StronglyEntanglingAllToAll(x.shape[1], n_layers)
    state = legacy_qg.angle_embedding(
        x,
        basis=basis,
        **legacy_kwargs,
    ).squeeze(-1)

    for layer_idx in range(n_layers):
        state = legacy_qg.apply_matrix(state, ansatz.layer_op(layer_idx, weights[layer_idx, 0]))

    return legacy_qg.measure(state, legacy_qg.sigma_Z_like(x=state))
