import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from .runtime import SpinRuntime


def _atoms_magmoms(atoms):
    mm = atoms.get_initial_magnetic_moments()
    mm = np.asarray(mm, dtype=float)
    if mm.ndim not in (1, 2) or (mm.ndim == 2 and mm.shape[1] != 3):
        raise ValueError(f"initial magnetic moments must be (N,) or (N, 3); got {mm.shape}")
    return mm


class SpinCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model_path, device="cuda", **kwargs):
        super().__init__(**kwargs)
        self.runtime = SpinRuntime(model_path, device=device)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        out = self.runtime.efs(atoms, _atoms_magmoms(atoms))
        self.results["energy"] = out["energy"]
        self.results["free_energy"] = out["energy"]
        self.results["forces"] = out["forces"]
        self.results["stress"] = full_3x3_to_voigt_6_stress(out["stress_3x3"])
