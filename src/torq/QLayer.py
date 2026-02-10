import torch
import torch.nn as nn
import torq as tq
from .Ansatz import make_ansatz
from ._pennylane_backend import maybe_create_pennylane_backend


class QLayer(nn.Module):
    def __init__(self, n_qubits=3, n_layers=1, ansatz_name="basic_entangling", config=None, weights=None,
                 weights_last_layer_data_re=None, q_layer_idx=0, param_init_dict=None, basis_angle_embedding='X'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.config = config
        self.ansatz_name = ansatz_name
        self.basis_angle_embedding = basis_angle_embedding

        # === data re-upload config ===
        self.data_reupload_every = getattr(self.config, "data_reupload_every", 0)  # 0 → off

        self.angle_scaling_method = getattr(self.config, 'angle_scaling_method', 'none')
        self.angle_scaling = getattr(self.config, 'angle_scaling', 1.0)

        self.reparametrize_sin_cos = getattr(self.config, 'reparametrize_sin_cos', False)
        # === ansatz selection + parameter tensor shape ===
        # build ansatz object (holds any precomputes)
        # NOTE: device of params is not known yet; we’ll move precomputes lazily on first forward if needed
        self.ansatz = make_ansatz(ansatz_name, n_qubits, n_layers, device=None)
        self.param_init(weights, weights_last_layer_data_re, param_init_dict, q_layer_idx)

        # sigma_Z observable like you had
        self.observable = tq.sigma_Z_like(x=self.params)  # keeps dtype/device consistent via likeable
        self._optional_backend = maybe_create_pennylane_backend(self)

    def forward(self, x):
        if not torch.isfinite(self.params).all():
            raise ValueError(f"QLayer.params has NaN: {self.params}")
        if self._optional_backend is not None:
            return self._optional_backend.forward(x)
        if not self.data_reupload_every:
            state = tq.angle_embedding(
                x,
                angle_scaling_method=self.angle_scaling_method,
                angle_scaling=self.angle_scaling,
                basis=self.basis_angle_embedding,
            ).squeeze(-1)  # [B,2**n]
            data_gates = None
        else:
            state = tq.get_initial_state(
                n_qubits=self.n_qubits,
                batch_size=x.shape[0],
                x=self.params,
            ).squeeze(-1)  # [B, 2**n]
            data_gates = tq.data_reuploading_gates(
                x,
                angle_scaling_method=self.angle_scaling_method,
                angle_scaling=self.angle_scaling,
                basis=self.basis_angle_embedding,
            )  # [B, n, 2, 2]

        angles_reparametrize = None
        if self.reparametrize_sin_cos:
            angles_reparametrize = torch.atan2(torch.sin(self.params), torch.cos(self.params))

        reps = max(self.data_reupload_every, 1)
        for layer in range(self.n_layers):
            for d_reup in range(reps):
                if self.reparametrize_sin_cos:
                    w = angles_reparametrize[layer, d_reup]
                else:
                    w = self.params[layer, d_reup]
                idx = d_reup if self.data_reupload_every else layer
                U = self.ansatz.layer_op(idx, w)  # [2**n, 2**n]
                state = tq.apply_matrix(state, U)

            if self.data_reupload_every:
                state = tq.data_reuploading(state, data_gates).squeeze(-1)  # [B,2**n]

        if self.data_reupload_every:
            angles_reparametrize_last = None
            if self.reparametrize_sin_cos:
                angles_reparametrize_last = torch.atan2(
                    torch.sin(self.params_last_layer_reupload),
                    torch.cos(self.params_last_layer_reupload),
                )
            for d_reup in range(self.data_reupload_every):
                if self.reparametrize_sin_cos:
                    w = angles_reparametrize_last[d_reup]
                else:
                    w = self.params_last_layer_reupload[d_reup]
                U = self.ansatz.layer_op(layer_idx=d_reup, weights=w)  # [2**n, 2**n]
                state = tq.apply_matrix(state, U)

        return tq.measure(state, self.observable)

    def param_init(self, weights=None, weights_last_layer_data_re=None, param_init_dict=None, layer_idx=0):
        with torch.no_grad():
            if weights is not None:
                parameters = weights
                if self.data_reupload_every:
                    parameters_reupload = (weights_last_layer_data_re)
            else:
                # resolve per-layer param shapes from ansatz
                if self.ansatz.param_shape[0] != "per_layer":
                    raise RuntimeError("Unexpected param_shape kind")
                per_layer = self.ansatz.param_shape[1]
                # Substitute n_qubits for Nones in the shape
                resolved = tuple(self.n_qubits if d is None else d for d in
                                 per_layer)  # shape: (n_qubits, n_params_per_layer)  (make sure it is correct

                param_scaling = 1.0 if self.reparametrize_sin_cos else 2 * torch.pi

                if getattr(self.config, 'init_identity', False):  # write the 3 fixed initializations better
                    parameters = torch.zeros(self.n_layers, max(self.data_reupload_every, 1), *resolved)
                    if self.data_reupload_every:
                        parameters_reupload = torch.zeros(self.data_reupload_every, *resolved)

                elif getattr(self.config, 'init_ones', False):
                    parameters = torch.pi * torch.ones(self.n_layers, max(self.data_reupload_every, 1), *resolved)
                    if self.data_reupload_every:
                        parameters_reupload = torch.pi * torch.ones(self.data_reupload_every, *resolved)

                elif getattr(self.config, 'init_pi_half', False):
                    parameters = (torch.pi / 2) * torch.ones(self.n_layers, max(self.data_reupload_every, 1), *resolved)
                    if self.data_reupload_every:
                        parameters_reupload = (torch.pi / 2) * torch.ones(self.data_reupload_every, *resolved)

                elif param_init_dict is not None:  # currently initializes all of the layers the same way and adding noise to the first QLayer. could be changed
                    match param_init_dict["numerator"]:
                        case "pi_range":
                            parameters = torch.pi * torch.rand(self.n_layers, max(self.data_reupload_every, 1), *resolved)
                            parameters_reupload = torch.pi * torch.rand(self.data_reupload_every, *resolved)
                        case "2pi_range":
                            parameters = torch.pi * 2 * (torch.rand(self.n_layers, max(self.data_reupload_every, 1),
                                                                      *resolved) - 0.5)
                            parameters_reupload = torch.pi * 2 * (torch.rand(self.data_reupload_every, *resolved) - 0.5)
                        case _:  # numerical value
                            init_bound = param_init_dict["numerator"] / param_init_dict["omega_0"]
                            parameters = init_bound * 2 * (torch.rand(self.n_layers, max(self.data_reupload_every, 1),
                                                                       *resolved) - 0.5)
                            parameters_reupload = init_bound * 2 * (torch.rand(self.data_reupload_every, *resolved) - 0.5)
                    noise_all_q_layers = getattr(self.config, 'noise_all_q_layers', False)
                    if (layer_idx == 1) or noise_all_q_layers:  # add noise
                        scale = param_init_dict["S1"] / param_init_dict["omega_0"]
                        noise = torch.randn_like(parameters) * scale
                        noise_reupload = torch.randn_like(parameters_reupload) * scale
                        parameters = parameters + noise
                        parameters_reupload = parameters_reupload + noise_reupload
                else:
                    parameters = param_scaling * torch.rand(self.n_layers, max(self.data_reupload_every, 1),
                                                            *resolved)
                    parameters_reupload = param_scaling * torch.rand(self.data_reupload_every, *resolved)

            self.params = nn.Parameter(parameters)
            if self.data_reupload_every:
                self.params_last_layer_reupload = nn.Parameter(parameters_reupload)


# class QLayer(nn.Module):
#     def __init__(self, n_qubits=3, n_layers=1, ansatz_name="basic_entangling", config=None,
#                  basis_angle_embedding='X'):
#         super().__init__()
#         self.n_qubits = n_qubits
#         self.n_layers = n_layers
#         self.config = config
#         self.ansatz_name = ansatz_name
#         self.basis_angle_embedding = basis_angle_embedding
#
#         self.angle_scaling_method = getattr(self.config, 'angle_scaling_method', 'none')
#         self.angle_scaling = getattr(self.config, 'angle_scaling', 1.0)
#
#         # === ansatz selection + parameter tensor shape ===
#         # build ansatz object (holds any precomputes)
#         # NOTE: device of params is not known yet; we’ll move precomputes lazily on first forward if needed
#         self.ansatz = make_ansatz(ansatz_name, n_qubits, n_layers, device=None)
#
#         # sigma_Z observable
#         self.observable = tq.sigma_Z_like(x=self.params)  # keeps dtype/device consistent via likeable
#         self._optional_backend = maybe_create_pennylane_backend(self)  # for benchmarking and testing against pennylane; if it is None, the regular forward will be used. for benchmarking, the TorQ-bench library should be used.
#
#     def forward(self, x):
#         if not torch.isfinite(self.params).all():
#             raise ValueError(f"QLayer.params has NaN: {self.params}")
#         if self._optional_backend is not None:
#             return self._optional_backend.forward(x)
#         state = tq.angle_embedding(
#             x,
#             angle_scaling_method=self.angle_scaling_method,
#             angle_scaling=self.angle_scaling,
#             basis=self.basis_angle_embedding,
#         ).squeeze(-1)  # [B,2**n]
#
#         angles_reparametrize = None
#         if self.reparametrize_sin_cos:
#             angles_reparametrize = torch.atan2(torch.sin(self.params), torch.cos(self.params))
#
#         for layer in range(self.n_layers):
#             if self.reparametrize_sin_cos:
#                 w = angles_reparametrize[layer]
#             else:
#                 w = self.params[layer]
#             U = self.ansatz.layer_op(layer, w)  # [2**n, 2**n]
#             state = tq.apply_matrix(state, U)
#
#         return tq.measure(state, self.observable)
