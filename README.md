# TorQ Quantum Library

TorQ is a lightweight, Torch-based statevector library for fast quantum layers inside neural networks.

Developed while writing the paper:
Quantum Physics-Informed Neural Networks for Maxwell's Equations: Circuit Design, "Black Hole" Barren Plateaus Mitigation, and GPU Acceleration

by Ziv Chen, Gal G. Shaviner, Hemanth Chandravamsi, Shimon Pisnoy, Steven H. Frankel, Uzi Pereg

https://arxiv.org/abs/2506.23246

For the simpler version associated with the following paper, please use the simple_for_PINNACLE branch:
PINNACLE: An Open-Source Computational Framework for Classical and Quantum PINNs

by Shimon Pisnoy, Hemanth Chandravamsi, Ziv Chen, Aaron Goldgewert, Gal Shaviner, Boris Shragner, Steven H. Frankel

https://arxiv.org/abs/2604.15645

It is compatible with TorQ-bench simple_for_PINNACLE branch and PINNACLE repos. Also, this branch is the same as torq-quantum v0.1.2 on PyPI.

## When to use

- You want a differentiable quantum layer inside a PyTorch model.
- You care about batched statevector throughput on CPU or GPU.
- You need analytic expectation values rather than shot-based sampling.

## When not to use

- You need realistic hardware noise models or error channels.
- You need shot-based measurements or sampling.
- You need hardware-aware compilation or execution on QPUs.
- You need multi-GPU or distributed statevector simulation.

## What this library optimizes for

- Speed inside a neural-network forward and backward pass.
- Batched evaluation of many inputs at once.
- Single-device training workflows.

This library was benchmarked primarily on NVIDIA L40s and A100 GPUs. Performance on other hardware has not been tuned and may differ.

## Install

TorQ requires Python 3.10+ and PyTorch 1.13+.

```bash
# install from PyPI - not fully updated yet
python -m pip install torq-quantum

# optional: editable install for local development
python -m pip install -e .
```

The PyPI package name is `torq-quantum`, while the Python import path is `torq`.

If you want to use the optional PennyLane comparison backend, you also need importable `pennylane` and `torq_bench` packages in the same environment.

## Quickstart

`torq.simple.Circuit` is the recommended student-facing API.

```python
import torch
from torq.simple import Circuit, CircuitConfig

circuit = Circuit(
    n_qubits=4,
    n_layers=2,
    ansatz_name="basic_entangling",
    config=CircuitConfig(),
)

x = torch.rand(8, 4)
y = circuit(x)

print(y.shape)  # torch.Size([8, 4])
```

By default, the output is the local Pauli-Z expectation value on each qubit, so the shape is `[batch, n_qubits]`.

## Student-facing API

- `Circuit`: `torch.nn.Module` wrapper around TorQ's internal `QLayer`.
- `CircuitConfig`: configuration object for encoding, ansatz options, measurements, and backend selection.
- `circuit.params`: trainable main parameter tensor.
- `circuit.params_last_layer_reupload`: extra trainable tensor used only when `data_reupload_every > 0`.

TorQ also exposes lower-level helpers through `torq.simple`:

```python
from torq.simple import (
    angle_embedding,
    data_reuploading,
    data_reuploading_gates,
    get_initial_state,
    measure,
)
```

## Supported ansatzes

TorQ currently supports these ansatz names:

- `basic_entangling`
- `single_rot_basic_ent`
- `strongly_entangling`
- `tile`
- `cross_mesh`
- `cross_mesh_2_rots`
- `cross_mesh_cx_rot`
- `no_entanglement_ansatz`

Notes:

- `single_rot_basic_ent` uses the same CNOT ladder as `basic_entangling`, but only one rotation parameter per qubit. The rotation axis is chosen with `single_rotation_gate`.
- `tile` uses a brick-wall CNOT pattern. Within each sublayer it applies `(0,1), (2,3), ...`, then `(1,2), (3,4), ...`. If `tile_cyclic=True` and `n_qubits` is even, it also adds a wraparound `CX(n_qubits - 1, 0)` per sublayer.
- `no_entanglement_ansatz` applies only single-qubit rotations.

Relevant `CircuitConfig` fields:

- `single_rotation_gate`: rotation axis used by `single_rot_basic_ent`, and by `tile` when `tile_rotation_params=1`. Supported values are `"x"`, `"rx"`, `"y"`, `"ry"`, `"z"`, `"rz"`.
- `tile_rotation_params`: `1` or `3`.
- `tile_sublayers`: number of repeated brick-wall sublayers per layer.
- `tile_cyclic`: whether `tile` adds the even-qubit wraparound CNOT.

Example: `tile`

```python
import torch
from torq.simple import Circuit, CircuitConfig

cfg = CircuitConfig(
    tile_rotation_params=1,
    single_rotation_gate="rz",
    tile_sublayers=2,
    tile_cyclic=True,
)

circuit = Circuit(
    n_qubits=6,
    n_layers=2,
    ansatz_name="tile",
    config=cfg,
)

x = torch.rand(8, 6)
y = circuit(x)
```

## Angle embedding and scaling

`CircuitConfig` controls angle embedding with these fields:

- `basis_angle_embedding`: `"x"`, `"rx"`, `"y"`, `"ry"`, `"z"`, or `"rz"`.
- `angle_scaling_method`: `"none"`, `"scale"`, `"scale_with_bias"`, `"asin"`, or `"acos"`.
- `angle_scaling`: scaling factor used by `"scale"` and `"scale_with_bias"`.

The nonlinear scaling modes are mainly intended for inputs that already live near `[-1, 1]`, for example after a `tanh`.

Scaling methods:

- `"none"`: leaves the input unchanged.
- `"scale"`: computes `angles * angle_scaling`.
- `"scale_with_bias"`: computes `(angles + 1) * (angle_scaling / 2)`.
- `"asin"`: computes `asin(angles) + pi / 2` after a small internal clamp for numerical stability.
- `"acos"`: computes `acos(angles)` after a small internal clamp for numerical stability.

```python
import torch
from torq.simple import Circuit, CircuitConfig

cfg = CircuitConfig(
    angle_scaling_method="scale",
    angle_scaling=torch.pi,
    basis_angle_embedding="Y",
)

circuit = Circuit(
    n_qubits=4,
    n_layers=2,
    ansatz_name="basic_entangling",
    config=cfg,
)

x = torch.rand(8, 4)
y = circuit(x)
```

## Data reuploading

Set `data_reupload_every > 0` to enable TorQ's legacy repeated-upload execution scheme.

When `data_reupload_every = k`:

- each logical layer applies `k` ansatz blocks before a data upload,
- after the final upload, TorQ applies one more tail of `k` ansatz blocks,
- the model therefore owns both `circuit.params` and `circuit.params_last_layer_reupload`.

This increases simulation cost and memory use.

```python
import torch
from torq.simple import Circuit, CircuitConfig

cfg = CircuitConfig(data_reupload_every=2)
circuit = Circuit(
    n_qubits=4,
    n_layers=2,
    ansatz_name="cross_mesh",
    config=cfg,
)

x = torch.rand(8, 4)
y = circuit(x)
```

Notes:

- For `strongly_entangling`, `data_reupload_every` must be `<= n_layers`.
- If you pass manual weights and `data_reupload_every == 0`, `weights` may be shaped as either `[n_layers, ...]` or `[n_layers, 1, ...]`.
- If you pass manual weights and `data_reupload_every > 0`, `weights` must have shape `[n_layers, data_reupload_every, ...]`, and `weights_last_layer_data_re` must have shape `[data_reupload_every, ...]`.

## Measurements and observables

Use `CircuitConfig(observables=...)` to control the output measurement.

Supported inputs:

- `None`: default local Pauli-Z on every qubit, returning `[batch, n_qubits]`.
- Pauli string: one or more Pauli words separated by `_`, case-insensitive.
- Hermitian matrix or matrices:
  - `[2, 2]` shared local observable for every qubit
  - `[n_qubits, 2, 2]` per-qubit local observables
  - `[2**n, 2**n]` one full-system observable
  - `[m, 2**n, 2**n]` multiple full-system observables

Pauli-string rules:

- Allowed characters are `I`, `X`, `Y`, `Z`, and `_`.
- A Pauli word of length `L` is measured on every contiguous `L`-qubit window.
- `_` concatenates multiple Pauli-word groups into one output tensor.
- An all-identity observable such as `"I_I_I"` is rejected.

Examples:

- `"z"` returns local `Z_i` on every qubit.
- `"xx"` returns `X_i X_{i+1}` on every adjacent pair.
- `"z_zz_x"` returns all `Z_i`, then all `Z_i Z_{i+1}`, then all `X_i`.

`pauli_measurement_chunk_size` tunes the throughput-versus-memory tradeoff for grouped Pauli-string measurements. Repeated all-`Z` words such as `"zz"` and `"zzz"` use a dedicated fast path.

```python
import torch
from torq.simple import Circuit, CircuitConfig

q = 4

cfg = CircuitConfig(
    observables="z_zz_x",
    pauli_measurement_chunk_size=8,
)

circuit = Circuit(
    n_qubits=q,
    n_layers=2,
    ansatz_name="basic_entangling",
    config=cfg,
)

x = torch.rand(8, q)
y = circuit(x)

print(y.shape)  # torch.Size([8, 11])
```

If local Pauli-Z is the only output you need, leaving `observables=None` is still the fastest path.

## Initialization and advanced options

Useful `CircuitConfig` flags:

- `init_identity=True`: initialize all parameters to `0`.
- `init_ones=True`: initialize all parameters to `pi`.
- `init_pi_half=True`: initialize all parameters to `pi / 2`.
- `reparametrize_sin_cos=True`: internally wraps angles through `atan2(sin(theta), cos(theta))`.
- `pennylane_backend=True`: try to run through the optional PennyLane comparison backend.
- `pennylane_dev_name`: PennyLane device name, defaulting to `"default.qubit"`.

`noise_all_q_layers` is only used by TorQ's legacy `param_init_dict` initialization path.

## Optional PennyLane backend

TorQ can optionally delegate execution to a PennyLane comparison backend:

```python
from torq.simple import Circuit, CircuitConfig

cfg = CircuitConfig(
    pennylane_backend=True,
    pennylane_dev_name="default.qubit",
)

circuit = Circuit(
    n_qubits=2,
    n_layers=1,
    ansatz_name="strongly_entangling",
    config=cfg,
)
```

Notes:

- TorQ only uses this path if both `pennylane` and `torq_bench` are importable.
- If the requested basis, observables, or ansatz are unsupported by the PennyLane comparison backend, TorQ emits a warning and falls back to the native TorQ backend.
- This path is mainly useful for parity checks and benchmarking rather than for normal TorQ usage.

## Benchmarking

For benchmark scripts and PennyLane comparisons, use the TorQ-bench repository:

https://github.com/zivchen9993/TorQ-bench

## Limitations

- Statevector simulation only, so memory scales as `O(2^n_qubits)`.
- Analytic measurements only; no shots or sampling.
- Ideal, noise-free gates and measurements.

## License

MIT. See `LICENSE`.
