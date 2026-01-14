import torch
import numpy as np
import torch.nn as nn
# import quantum_lib.QGates as qg
# import device_and_dtype as dev
from utility import PennyLaneSanityCheck as qml
import quantum_lib as qg
from quantum_lib.Ansatz import make_ansatz
import matplotlib.pyplot as plt


class QLayer(nn.Module):
    def __init__(self, n_qubits=3, n_layers=1, ansatz_name="strongly_entangling", config=None, weights=None,
                 weights_last_layer_data_re=None, q_layer_idx=0, param_init_dict=None, basis_angle_embedding='X'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.config = config
        self.ansatz_name = ansatz_name
        self.basis_angle_embedding = basis_angle_embedding

        # === data re-upload config ===
        self.data_reupload_every = getattr(self.config, "data_reupload_every", 0)  # 0 → off
        self.data_reuploading_gates = None

        self.angle_scaling = getattr(self.config, 'angle_scaling', 1.0)
        self.angle_asin = getattr(self.config, 'angle_asin', False)
        self.angle_acos = getattr(self.config, 'angle_acos', False)
        self.scale_with_bias = getattr(self.config, 'scale_with_bias', False)
        self.reparametrize_sin_cos = getattr(self.config, 'reparametrize_sin_cos', False)
        # === ansatz selection + parameter tensor shape ===
        # build ansatz object (holds any precomputes)
        # NOTE: device of params is not known yet; we’ll move precomputes lazily on first forward if needed
        self.ansatz = make_ansatz(ansatz_name, n_qubits, n_layers, device=None)
        self.param_init(weights, weights_last_layer_data_re, param_init_dict, q_layer_idx)

        # sigma_Z observable like you had
        self.observable = qg.sigma_Z_like(x=self.params)  # keeps dtype/device consistent via likeable

        self.pennylane_backend = getattr(self.config, 'pennylane_backend', False)
        if self.pennylane_backend:
            self.penny = qml.qml_sanity_check(n_qubits=n_qubits, n_layers=n_layers, weights=self.params,
                                              pennylane_dev_name=getattr(self.config, 'pennylane_dev_name', False))
            self.qc = self.penny.circuit_strongly_entangling()

    def forward(self, x):
        ####### Debug ########
        if not torch.isfinite(self.params).all():
            raise ValueError(f"QLayer.params has NaN: {self.params}")
        ########## check pennylane timing ##########

        ######################
        if self.pennylane_backend:
            x = self.angle_scaler(x)  # scale input angles to [0, 2π]
            return torch.stack(self.qc(x), dim=1).to(
                torch.float32)  # [B, 2**n]  # for debugging purposes, comparing to pennylane

        else:
            if not self.data_reupload_every:
                state = qg.angle_embedding(x, scale=self.angle_scaling, asin=self.angle_asin, acos=self.angle_acos,
                                           basis=self.basis_angle_embedding,
                                           scale_with_bias=self.scale_with_bias).squeeze(-1)  # [B,2**n]
            else:
                # state = qg.get_initial_state(n_qubits=self.n_qubits, x=self.params[0]).squeeze(-1)  # [B,2**n]
                state = qg.get_initial_state(
                    n_qubits=self.n_qubits,
                    batch_size=x.shape[0],  # <- match B
                    x=self.params  # use @likeable to pick dtype/device
                ).squeeze(-1)  # [B, 2**n]

                self.data_reuploading_gates, B, n_qubits = qg.data_reuploading_gates(x, scale=self.angle_scaling,
                                                                                     asin=self.angle_asin,
                                                                                     acos=self.angle_acos,
                                                                                     scale_with_bias=self.scale_with_bias,
                                                                                     basis=self.basis_angle_embedding)  # [B, n, 2, 2]
            # 2) for each layer: build full 2^n×2^n and apply it
            for layer in range(self.n_layers):
                for d_reup in range(
                        max(self.data_reupload_every, 1)):  # 1 to make it work when data-reuploading is set to 0
                    if self.reparametrize_sin_cos:
                        angles_reparametrize = torch.atan2(torch.sin(self.params), torch.cos(self.params))
                        w = angles_reparametrize[layer, d_reup]
                    else:
                        w = self.params[layer, d_reup]
                    idx = d_reup if self.data_reupload_every else layer
                    # U = self.ansatz.layer_op(layer, w)  # [2**n, 2**n]
                    U = self.ansatz.layer_op(idx, w)  # [2**n, 2**n]  # d_reup because that is the restarting point for another call to qml.StronglyEnt()
                    state = qg.apply_matrix(state, U)
                    # if hasattr(self.ansatz, "apply"):
                    #     # build data wall once per re-upload point (if enabled)
                    #     state = self.ansatz.apply(state, layer, w, idx_pairs=self.idx_pairs)
                    # else:
                    #     U = self.ansatz.layer_op(layer, w)  # [2**n, 2**n]
                    #     state = qg.apply_matrix(state, U)

                if self.data_reupload_every:
                    # print(f"entered data_reupload for the {d_reup} time in the {layer} layer")
                    state = qg.data_reuploading(B, n_qubits, state, self.data_reuploading_gates).squeeze(-1)  # [B,2**n]

            if self.data_reupload_every:
                for d_reup in range(self.data_reupload_every):
                    if self.reparametrize_sin_cos:
                        angles_reparametrize = torch.atan2(torch.sin(self.params_last_layer_reupload),
                                                           torch.cos(self.params_last_layer_reupload))
                        w = angles_reparametrize[d_reup]
                    else:
                        w = self.params_last_layer_reupload[d_reup]
                    U = self.ansatz.layer_op(layer_idx=d_reup, weights=w)  # [2**n, 2**n]
                    state = qg.apply_matrix(state, U)

                # state = qg.apply_matrix(state, U)

            # if self.config.count_entanglement:
            #     if self.config.global_step % self.config.print_idx == 0 or self.config.global_step == self.config.epochs - 1:
            #         Q_lin, Q_R2 = qg.global_entanglement_calc(
            #             state)  # global (Meyer–Wallach) entanglement (linear), and R2 (Renyi-2) entanglement
            #         self.config.writer.add_scalar("entanglement/linear_global_entanglement", Q_lin,
            #                                       self.config.global_step)
            #         self.config.writer.add_scalar("entanglement/R2_global_entanglement", Q_R2, self.config.global_step)
            # 3) measure
            return qg.measure(state, self.observable)

    def angle_scaler(self, angles):
        """
        Scale input angles to [0, 2π] if angle_scaling is set.
        """
        if self.angle_acos:
            angles = torch.acos(angles)
        elif self.angle_asin:
            angles = torch.asin(angles) + (torch.pi / 2)
            # if scale_with_bias:
            #     angles = torch.asin(angles) + (torch.pi / 2)
            # else:
            #     angles = torch.asin(angles)
        elif self.angle_scaling is not None:  # can't work with both scaling and arcsin
            if self.scale_with_bias:
                angles = (angles + 1) * (self.angle_scaling / 2)  # scale to [0, scale] - which means [0, pi]
            else:
                angles = angles * self.angle_scaling
        return angles

    def param_init(self, weights=None, weights_last_layer_data_re=None, param_init_dict=None, layer_idx=0, pi_range=False):
        with torch.no_grad():
            if weights is not None:
                # self.params.copy_(weights)
                # if self.data_reupload_every:
                #     self.params_last_layer_reupload.copy_(weights_last_layer_data_re)
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

                if getattr(self.config, 'init_identity', False):  # TODO: write the 3 fixed initializations better
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

                elif param_init_dict is not None:  # TODO: currently initializes all of the layers the same way and adding noise to the first QLayer. could be changed
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
                    # self.plot_tensor_distribution(parameters, density=True, title=f"params ({str(param_init_dict['numerator'])})")
                    # self.plot_tensor_distribution(parameters_reupload, density=True, title=f"parameters_reupload ({str(param_init_dict['numerator'])})")
                    noise_all_q_layers = getattr(self.config, 'noise_all_q_layers', False)
                    if (layer_idx == 1) or noise_all_q_layers:  # add noise
                        scale = param_init_dict["S1"] / param_init_dict["omega_0"]
                        noise = torch.randn_like(parameters) * scale
                        noise_reupload = torch.randn_like(parameters_reupload) * scale
                        parameters = parameters + noise
                        parameters_reupload = parameters_reupload + noise_reupload
                        # self.plot_tensor_distribution(noise, density=True, title=f"noise ({str(param_init_dict['numerator'])})")
                        # self.plot_tensor_distribution(noise_reupload, density=True, title=f"noise_reupload ({str(param_init_dict['numerator'])})")
                        # self.plot_tensor_distribution(parameters, density=True, title=f"sum params + noise ({str(param_init_dict['numerator'])})")
                        # self.plot_tensor_distribution(parameters_reupload, density=True, title=f"sum parameters_reupload + noise_reupload ({str(param_init_dict['numerator'])})")
                else:
                    parameters = param_scaling * torch.rand(self.n_layers, max(self.data_reupload_every, 1),
                                                            *resolved)
                    parameters_reupload = param_scaling * torch.rand(self.data_reupload_every, *resolved)

            self.params = nn.Parameter(parameters)
            if self.data_reupload_every:
                self.params_last_layer_reupload = nn.Parameter(parameters_reupload)

    def plot_tensor_distribution(self, t, bins="auto", density=False, logy=False, title=None):  # for debugging
        # make it 1D on CPU as float
        vals = t.detach().flatten().float().cpu().numpy()

        plt.figure()
        plt.hist(vals, bins=bins, density=density)
        if logy:
            plt.yscale("log")
        if title:
            plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Density" if density else "Count")
        plt.tight_layout()
        plt.show()

# if __name__ == "__main__":
#     device = dev.device
#     dtype = dev.dtype
#     torch.set_default_dtype(dev.dtype)
#     n_layers = 3
#     n_qubits = 5
#
#     single = True
#     batch = True
#     # zero_weights = True
#     zero_weights = False
#     # For testing, create some input angles.
#     # Here, we assume inputs is now batch-aware.
#     # For example, a batch of 4 samples each with n_qubits angles:
#     inputs_batch = torch.rand(4, n_qubits).to(device)  # shape [B, n_qubits]
#     # inputs_batch = torch.zeros(4, n_qubits).to(device)  # shape [B, n_qubits]
#     inputs_single = inputs_batch[0].detach().clone().to(device)  # shape [B, n_qubits]
#     # The weights for the entangling layers remain shared (shape [n_layers, n_qubits, 3]).
#     # weights = torch.rand(n_layers, n_qubits, 3)
#     # weights = torch.rand(n_layers, n_qubits + n_qubits ** 2)
#     # weights = torch.zeros(n_layers, n_qubits, 3)
#
#     # Instantiate the quantum layer.
#     # q_layer_old = ql.QLayer(n_qubits=n_qubits, n_layers=n_layers, weights=weights)
#     for ans in range(5):
#         if ans == 0:
#             if zero_weights:
#                 weights = torch.zeros(n_layers, n_qubits ** 2)
#             else:
#                 weights = torch.rand(n_layers, n_qubits ** 2)
#             penny = qml.qml_sanity_check(n_qubits=n_qubits, n_layers=n_layers, weights=weights)
#             ansatz_name = "cross_mesh"
#             qc = penny.circuit_cross_mesh()
#         elif ans == 1:
#             if zero_weights:
#                 weights = torch.zeros(n_layers, n_qubits + n_qubits ** 2)
#             else:
#                 weights = torch.rand(n_layers, n_qubits + n_qubits ** 2)
#             ansatz_name = "cross_mesh_2_rots"
#             penny = qml.qml_sanity_check(n_qubits=n_qubits, n_layers=n_layers, weights=weights)
#             qc = penny.circuit_cross_mesh_2_rots()
#         elif ans == 2:
#             if zero_weights:
#                 weights = torch.zeros(n_layers, n_qubits, 3)
#             else:
#                 weights = torch.rand(n_layers, n_qubits, 3)
#             ansatz_name = "cross_mesh_cx_rot"
#             penny = qml.qml_sanity_check(n_qubits=n_qubits, n_layers=n_layers, weights=weights)
#             qc = penny.circuit_cross_mesh_cx_rot()
#         elif ans == 3:
#             if zero_weights:
#                 weights = torch.zeros(n_layers, n_qubits, 3)
#             else:
#                 weights = torch.rand(n_layers, n_qubits, 3)
#             penny = qml.qml_sanity_check(n_qubits=n_qubits, n_layers=n_layers, weights=weights)
#             ansatz_name = "strongly_entangling"
#             qc = penny.circuit_strongly_entangling()
#         elif ans == 4:
#             if zero_weights:
#                 weights = torch.zeros(n_layers, n_qubits, 3)
#             else:
#                 weights = torch.rand(n_layers, n_qubits, 3)
#             penny = qml.qml_sanity_check(n_qubits=n_qubits, n_layers=n_layers, weights=weights)
#             ansatz_name = "strongly_entangling_all_to_all"
#             qc = penny.circuit_strongly_entangling_all_to_all()
#         else:
#             raise ValueError("Invalid ansatz name")
#         q_layer = QLayer(n_qubits=n_qubits, n_layers=n_layers, weights=weights, ansatz_name=ansatz_name)
#         # Test the circuit.
#         inputs_batch.requires_grad_(True)
#         inputs_single.requires_grad_(True)
#         if single:
#             tensor_result_single = q_layer.forward(inputs_single)
#             # old_tensor_result_single = q_layer_old.forward(inputs_single)
#             pennylane_result_single = torch.tensor(qc(inputs_single))
#             print("Single ", ansatz_name, ": ",
#                   torch.sum(torch.abs(tensor_result_single - pennylane_result_single)).item())
#         if batch:
#             tensor_result_batch = q_layer.forward(inputs_batch)
#             # old_tensor_result_batch = q_layer_old.forward(inputs_batch)
#             pennylane_result_batch = torch.tensor(qc(inputs_batch[0])).unsqueeze(0)
#             for i in range(1, len(inputs_batch)):
#                 pennylane_result_batch_temp = torch.tensor(qc(inputs_batch[i])).unsqueeze(0)
#                 pennylane_result_batch = torch.cat((pennylane_result_batch, pennylane_result_batch_temp), dim=0)
#             print("Batch ", ansatz_name, ": ",
#                   torch.sum(torch.abs(pennylane_result_batch - tensor_result_batch)).item())
