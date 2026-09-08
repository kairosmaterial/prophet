# Prophet

Equivariant machine-learning interatomic potential (MLIP) by Kairos Materials: an
equivariant message-passing network (62.3M parameters, 10 layers, lmax 4, dual cutoffs
3/7 A) predicting energies, forces and stresses, exposed as an ASE calculator. This
repository contains the minimal inference code for the Matbench Discovery submission.

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

```python
from ase.build import bulk
from prophet import KairosCalculator

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = KairosCalculator(model_path="checkpoint.pt")
print(atoms.get_potential_energy())
print(atoms.get_forces())
print(atoms.get_stress())
```

`use_kernel=False` selects the pure e3nn tensor-product path and also runs on CPU.

## Checkpoint

The submission checkpoint (custom format: one-line JSON config header followed by the
PyTorch state dict) will be published on the Kairos Materials Hugging Face organization
and is loaded via `prophet.load_model(path)` or directly by `KairosCalculator`.

## Acknowledgements

Portions of the network utilities are adapted from e3nn, fairchem (eSEN) and
pytorch_runstats.

## License

MIT
