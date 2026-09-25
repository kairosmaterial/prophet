# Prophet

Atomistic foundation models for materials simulation and discovery.

Prophet is a family of atomistic foundation models developed by Kairos Materials. Trained on large-scale first-principles calculations—simulations based on quantum mechanics—Prophet predicts energies, forces and stresses from atomic structures. These predictions support the search for stable materials and simulations of atomic motion, without repeating an expensive first-principles calculation at every step.

First-principles calculations can describe materials that have never been synthesized, providing a way to learn beyond what is already known experimentally. Prophet learns from this computational experience through large-scale pretraining on [OMat24](https://omat24.org/) and [ELEMENTA](https://huggingface.co/datasets/kairosmaterial/ELEMENTA). Its development brings together broader training data and models that preserve the physical symmetries and smooth energy landscapes needed for atomistic simulation.

Prophet-Spin extends this approach by treating spin as an explicit physical degree of freedom. Represented through local magnetic moments, this additional information allows the model to distinguish different magnetic states of the same atomic structure and describe how they affect its energy and forces.

**Research preview.** All released models are preview versions for research purposes only. They are provided “as is”, without warranties or guarantees of accuracy, reliability or fitness for any particular purpose.

For access to more capable Prophet models, contact research@kairosmaterials.com.

## Research preview releases

This repository provides two research preview models with different purposes: a checkpoint aligned for Matbench Discovery evaluation, and a separate model with explicit spin inputs.

### Released models

#### Prophet-OAME-MBD — Matbench Discovery checkpoint

Prophet-OAME-MBD is the primary checkpoint used for our Matbench Discovery submission. It predicts energies, forces and stresses from atomic structures.

This checkpoint has been fine-tuned on MPtrj and sAlex (Alexandria) to align its energy predictions with the energy reference used by Matbench Discovery. Use this checkpoint, together with the corresponding submission protocol, when reproducing our benchmark results.

The energy-reference alignment is specific to this checkpoint. Other Prophet models should not be assumed to produce directly interchangeable energies or reproduce the same benchmark results.

**Checkpoint:** `prophet-oame-mbd.pt`

#### Prophet-Spin — Spin-dependent atomistic modeling

Prophet-Spin is a separate model that describes materials through both their atomic structure and their spin configuration. The released checkpoint is named Prophet-V1-Mag.

The same arrangement of atoms can support different spin configurations with different energies. Prophet-Spin represents this dependence explicitly as `E(R, M)`, where `R` denotes atomic positions and `M` denotes local magnetic moments. It predicts energies, forces and stresses for a specified magnetic configuration, supporting comparisons between magnetic states and studies of their coupling to atomic structure.

The release also includes a magnetic-moment prediction component for estimating local moments from a crystal structure. This is distinct from evaluating energies and forces at a user-specified magnetic configuration.

**Checkpoint:** `prophet-v1-mag.pt`

**Associated magnetic-moment prediction bundle:** `prophet-v1-mag-head.pt`

The prediction bundle belongs to the Spin release; it is not a third model. Prophet-V1-Mag is not the checkpoint used for the Matbench Discovery submission.

Checkpoints are distributed through the [Hugging Face model repository](https://huggingface.co/kairosmaterial/prophet). Each file uses a one-line JSON configuration header followed by a PyTorch state dict and can be loaded with `prophet.load_model(path)` or directly through the calculators.

## Installation

```bash
pip install "prophet-mlip @ git+https://github.com/kairosmaterial/prophet.git"
```

With the CUDA tensor-product kernels (recommended, requires a CUDA toolchain with
`nvcc` and headers at runtime for JIT compilation):

```bash
pip install "prophet-mlip[kernel] @ git+https://github.com/kairosmaterial/prophet.git"
```

## Quick start

### Structure energy, forces and stress

This example uses the `prophet-oame-mbd.pt` checkpoint.

```python
from ase.build import bulk
from prophet import KairosCalculator

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = KairosCalculator(
    model_path="prophet-oame-mbd.pt",
    use_kernel=False,
    device="cpu",
)
print(atoms.get_potential_energy())
print(atoms.get_forces())
print(atoms.get_stress())
```

`use_kernel=False` selects the pure e3nn tensor-product path and also runs on CPU.

### Energies and forces at a specified spin configuration

This is an independent magnetic example. The number of initial magnetic moments must match the number of atoms.

```python
from ase.build import bulk
from prophet_spin import SpinCalculator

atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
atoms.set_initial_magnetic_moments([2.5, -2.5])
atoms.calc = SpinCalculator(model_path="prophet-v1-mag.pt", device="cpu")
print(atoms.get_potential_energy())
print(atoms.get_forces())
print(atoms.get_stress())
```

The magnetic state is read from ASE's standard initial-magnetic-moments channel. Collinear `(N,)` and non-collinear `(N, 3)` moment arrays are supported.

### Magnetic-moment prediction

This independent example uses the prediction bundle associated with Prophet-Spin. Seed moments encode the starting magnetic ordering.

```python
from ase.build import bulk
from prophet_spin import MagmomPredictor

atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
atoms.set_initial_magnetic_moments([2.5, -2.5])
atoms.calc = MagmomPredictor(
    model_path="prophet-v1-mag.pt",
    head_path="prophet-v1-mag-head.pt",
    device="cpu",
)
print(atoms.get_magnetic_moments())
```

## Model details and evaluation

### Training

Prophet is pretrained on OMat24 and ELEMENTA. Released checkpoints then receive task-specific follow-up training: Prophet-OAME-MBD is fine-tuned on MPtrj and sAlex to align with the energy reference used by Matbench Discovery, while Prophet-V1-Mag is trained for spin-dependent energies, forces and stresses; its associated head bundle provides magnetic-moment prediction.

### Architecture

The Multi-Cutoff Spectral Decomposition (MCSD) represents interactions at several distance scales, combining short-range resolution with a wider radial field. The network preserves E(3) symmetry and a twice-continuously-differentiable (`C²`) energy, which supports smooth forces and stresses for atomistic simulation.

For implementation background, see the [Prophet technical report](https://www.kairosmaterials.com/papers/Prophet.pdf).

| Checkpoint | Parameters | Layers | Angular order | Interaction cutoffs | Symmetry | Energy smoothness |
|---|---:|---:|---:|---:|---|---|
| `prophet-oame-mbd.pt` | 62.3M | 10 | `lmax = 4` | 3 Å and 7 Å | E(3) equivariant | C² |
| `prophet-v1-mag.pt` | See checkpoint config | See checkpoint config | See checkpoint config | See checkpoint config | E(3) equivariant | C² |

The complete architecture configuration is stored in the JSON header of each checkpoint. The `prophet-v1-mag-head.pt` file is a prediction bundle containing the Prophet-Spin backbone, magnetic-moment head and seed data; it is not a third model.

### Matbench Discovery: `prophet-oame-mbd.pt`

Matbench Discovery results apply specifically to `prophet-oame-mbd.pt`. Reproduction should use the Matbench Discovery evaluation release and official submission protocol cited by that benchmark, together with this checkpoint's energy-reference handling. Report the evaluation version, protocol and any energy processing alongside the score. This repository does not add a separate benchmark score table; use the official submission result when comparing scores. The checkpoint is aligned with the energy reference used by Matbench Discovery; this does not mean it was trained on the benchmark test set.

### Spin evaluation: `prophet-v1-mag.pt`

Prophet-Spin results are separate from Matbench Discovery results and apply to `prophet-v1-mag.pt` under the stated magnetic configuration and evaluation procedure. Magnetic-moment results apply to the associated `prophet-v1-mag-head.pt` bundle. Do not interpret either preview's results as an upper bound on the capability of the Prophet model family.

## Limitations and intended use

The released checkpoints are intended for research and evaluation, not production deployment. Their performance may vary across chemical systems, structures and simulation conditions. Benchmark results apply to the checkpoint and evaluation procedure used and do not establish accuracy for every application.

Users are responsible for independently validating predictions and simulation results for their intended research. Magnetic-moment predictions should likewise be assessed for the system of interest and should not be treated as a guarantee of finding the magnetic ground state.

The research-preview designation describes release status and support expectations; it does not modify the licenses. Model weights remain available under CC-BY-4.0, including commercial use subject to the license terms, and the source code remains available under the MIT License. If a future release needs a different weight license or a transition arrangement, that change must be made in the license terms and reviewed separately; a README disclaimer cannot withdraw rights already granted under CC-BY-4.0.

## License

Source code is released under the MIT License (see [LICENSE](LICENSE)). Model weights distributed with releases are licensed under CC-BY-4.0 (see [LICENSE-WEIGHTS](LICENSE-WEIGHTS)). The two licenses apply separately. Third-party code included here retains its original license (see [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY)).

## Acknowledgements

Portions of the model implementation are derived from
[Nequix](https://github.com/atomicarchitects/nequix), used under the MIT
License (see [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY)).

Other network utilities are adapted from e3nn, fairchem (eSEN) and
pytorch_runstats.

Copyright (c) 2026 Kairos Materials
