# Prophet

Prophet is a family of large-scale pretrained foundation models for atomistic
simulation by Kairos Materials. Built on Multi-Cutoff Spectral Decomposition (MCSD),
Prophet models atomic interactions across nested cutoff scales, increasing the radial bandwidth 
of the network while preserving exact E(3) equivariance and a twice-continuously-differentiable energy. 
Prophet is pretrained on OMat24 and the composition-complete [ELEMENTA](https://huggingface.co/datasets/kairosmaterial/ELEMENTA)
dataset, with staged fine-tuning on MPtrj, sAlex and the ELEMENTA expansion sets.

Highlights:

- Universal energy/force/stress prediction across the periodic table, exposed as an
  ASE calculator.
- Sharp short-range resolution from the multi-cutoff design
- A magnetic extension, Prophet-Spin, makes the magnetic state an explicit input
  E(R, M). 

Technical report: [Prophet](https://www.kairosmaterials.com/papers/Prophet.pdf)

## Open-source releases

| Model | Task | Notes |
|---|---|---|
| Prophet-OAME-MBD | Energy / forces / stress | 62.3M params, 10 layers, lmax 4, cutoffs 3/7 A; thermal-conductivity fine-tuned; the Matbench Discovery submission checkpoint |
| Prophet-V1-Mag | E(R, M), forces, stress and magnetic-moment prediction | Prophet-Spin, Heisenberg-mode magnetic extension; collinear (N,) and vector (N,3) moment inputs |

Larger and more capable Prophet variants are available for commercial use; contact
research@kairosmaterials.com.

## Installation

```bash
pip install "prophet-mlip @ git+https://github.com/kairosmaterial/prophet.git"
```

With the CUDA tensor-product kernels (recommended, requires a CUDA toolchain with
`nvcc` and headers at runtime for JIT compilation):

```bash
pip install "prophet-mlip[kernel] @ git+https://github.com/kairosmaterial/prophet.git"
```

## Usage

Energy / forces / stress (Prophet-V1-MBD):

```python
from ase.build import bulk
from prophet import KairosCalculator

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = KairosCalculator(model_path="prophet-oame-mbd.pt")
print(atoms.get_potential_energy())
print(atoms.get_forces())
print(atoms.get_stress())
```

`use_kernel=False` selects the pure e3nn tensor-product path and also runs on CPU.

Magnetically conditioned EFS (Prophet-V1-Mag): energies, forces and stresses at a
fixed magnetic configuration. The magnetic state is read from the standard ASE
channel - collinear `(N,)` or non-collinear `(N, 3)` initial magnetic moments.

```python
from prophet_spin import SpinCalculator

atoms.set_initial_magnetic_moments([2.5, -2.5, 0, 0])
atoms.calc = SpinCalculator(model_path="prophet-v1-mag.pt")
print(atoms.get_potential_energy())
```

Magnetic-moment prediction (Prophet-V1-Mag): drive seed moments to self-consistency
and predict the magnetic ground state from the crystal structure alone. Predicted
moments are exposed through the standard ASE results channel.

```python
from prophet_spin import MagmomPredictor

atoms.calc = MagmomPredictor(model_path="prophet-v1-mag.pt", head_path="prophet-v1-mag-head.pt")
print(atoms.get_magnetic_moments())
```

## Checkpoints

Checkpoints are distributed via the releases of this repository (custom format:
one-line JSON config header followed by the PyTorch state dict) and load via
`prophet.load_model(path)` or directly through the calculators:

- `prophet-oame-mbd.pt` - Prophet-V1-MBD, the Matbench Discovery submission checkpoint
- `prophet-v1-mag.pt` - Prophet-V1-Mag energy/forces/stress model E(R, M)
- `prophet-v1-mag-head.pt` - Prophet-V1-Mag magnetic-moment prediction bundle
  (backbone, moment head and per-element seed table in one file)

## License

Source code is released under the MIT License (see LICENSE). Model weights
distributed via releases are licensed under CC-BY-4.0 (see LICENSE-WEIGHTS).

## Acknowledgements

Portions of the network utilities are adapted from e3nn, fairchem (eSEN) and
pytorch_runstats.
