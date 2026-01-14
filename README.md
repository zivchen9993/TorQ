# QPINNACLE Quantum Library

Lightweight, Torch-based statevector utilities built for fast quantum layers inside neural networks.

## When to use

- You want a fast, differentiable quantum layer inside a PyTorch model (`QLayer`/`Circuit`).
- You train with large batches on GPU and care about throughput.
- You need ideal (noise-free) analytic expectation values, not shot-based measurements. The noise and shots are features planned to be added in the future.

## When not to use

- You need realistic hardware noise models or error channels.
- You need shot-based measurements or sampling.
- You need hardware-specific circuit compilation.

## What this library optimizes for

- Speed as part of a neural network forward/backward pass.
- Batched, parallel evaluation (many inputs at once).
- Single-GPU execution for training pipelines.

This library was optimized and benchmarked primarily on single NVIDIA L40s and A100 GPUs.
Performance on other GPUs has not been tuned and may differ. Multi-GPU usage is not
currently available.

## Limitations

- Statevector simulation only (memory scales as O(2^n)).
- Analytic measurements only; no sampling/shots.
- Ideal, noise-free gates and measurements.

## Install

```bash
pip install -e .
```

## Quickstart

```python
import torch
from TorQ.simple import Circuit, CircuitConfig

n_qubits = 4
n_layers = 2

cfg = CircuitConfig(data_reupload_every=0)
circuit = Circuit(n_qubits=n_qubits, n_layers=n_layers, ansatz_name="strongly_entangling", config=cfg)

x = torch.rand(8, n_qubits)
y = circuit(x)
```

## Angle scaling example: (intended for usage with a tanh activation function before the quantum layer).

```python
import torch
from TorQ.simple import Circuit, CircuitConfig

cfg = CircuitConfig(angle_scaling=torch.pi, scale_with_bias=False)
circuit = Circuit(n_qubits=4, n_layers=2, ansatz_name="strongly_entangling", config=cfg)

x = torch.rand(8, 4)
y = circuit(x)
```

## Data reuploading

```python
import torch
from TorQ.simple import Circuit, CircuitConfig

cfg = CircuitConfig(data_reupload_every=2)
circuit = Circuit(n_qubits=4, n_layers=2, ansatz_name="cross_mesh", config=cfg)

x = torch.rand(8, 4)
y = circuit(x)
```

## Available ansatz names

- strongly_entangling
- strongly_entangling_all_to_all
- cross_mesh
- cross_mesh_2_rots
- cross_mesh_cx_rot
- no_entanglement_ansatz

## Available scalings names (intended for usage with a tanh activation function before the quantum layer)

- acos: acos(angles) -> output in [0, pi]
- asin: asin(angles) + pi/2 -> output in [0, pi]
- scale_with_bias(scale): (angles + 1) * (scale/2) -> output in [0, pi]  # usually scale is set to pi
- scale(scale): angles * scale -> output in [-pi, pi]  # usually scale is set to pi
- None: angles

## Low-level functions

```python
from TorQ.simple import (
    angle_embedding,
    data_reuploading_gates,
    data_reuploading,
    get_initial_state,
    measure,
)
```

## License

TBD
