import io
import json

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers
from ase.stress import full_3x3_to_voigt_6_stress

from .adapter import ProphetSpin
from .graph import Batch, Collator, sample_from_atoms
from .heads import MagmomHead, MagmomHeadX, features, features_ex
from .runtime import SpinRuntime


def load_head_bundle(path: str):
    with open(path, "rb") as f:
        config = json.loads(f.readline().decode())
        payload = torch.load(io.BytesIO(f.read()), map_location="cpu")
    backbone = ProphetSpin(config)
    state = {k: v for k, v in payload["backbone"].items() if ".tp." not in k}
    backbone.load_state_dict(state, strict=False)
    backbone.eval()
    if config.get("headx"):
        head = MagmomHeadX(e_max=float(config.get("e_max", 1.0)))
    else:
        head = MagmomHead()
    head.load_state_dict(payload["head"])
    head.eval()
    lut = torch.full((119, 3), 0.1)
    for sym, levels in config["seed_levels"].items():
        z = atomic_numbers.get(sym, 0)
        if 0 < z < 119:
            lut[z] = torch.tensor([float(x) for x in levels])
    return backbone, head, lut


class MagmomPredictor(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress", "magmoms"]

    def __init__(self, model_path, head_path, device="cuda", seed_mode="lut",
                 magnetic_threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        if seed_mode not in ("lut", "raw"):
            raise ValueError(f"seed_mode must be 'lut' or 'raw'; got {seed_mode!r}")
        self.seed_mode = seed_mode
        self.magnetic_threshold = float(magnetic_threshold)
        self.runtime = SpinRuntime(model_path, device=device)
        self.dev = self.runtime.dev
        self.backbone, self.head, self.lut = load_head_bundle(head_path)
        self.backbone = self.backbone.to(self.dev)
        self.head = self.head.to(self.dev)
        self.lut = self.lut.to(self.dev)
        self.collator = Collator(cutoff=self.backbone.graph_cutoff)

    @torch.no_grad()
    def predict_moments(self, atoms, seeds):
        seeds = np.asarray(seeds, dtype=float)
        if seeds.ndim != 1:
            raise ValueError(
                f"the moment head is collinear: seeds must be (N,); got {seeds.shape}"
            )
        m = torch.as_tensor(seeds, dtype=torch.float32, device=self.dev)
        if self.seed_mode == "lut":
            z = torch.as_tensor(
                atoms.get_atomic_numbers(), dtype=torch.long, device=self.dev
            )
            magnetic = m.abs() > self.magnetic_threshold
            m0 = torch.where(
                magnetic,
                torch.sign(m) * self.lut[z.clamp(max=118), 1],
                torch.zeros_like(m),
            )
        else:
            m0 = m
        sample = sample_from_atoms(atoms, m0.cpu().numpy())
        batch = Batch(self.collator([sample])).to(self.dev)
        if isinstance(self.head, MagmomHeadX):
            h, e_site = features_ex(self.backbone, batch, batch.magmoms)
            pred = self.head(batch.magmoms, h, e_site)
        else:
            pred = self.head(batch.magmoms, features(self.backbone, batch, batch.magmoms))
        return pred.cpu().numpy()

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        seeds = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
        if not np.any(seeds):
            raise ValueError(
                "MagmomPredictor needs seed moments encoding the magnetic ordering: "
                "call atoms.set_initial_magnetic_moments(...) first"
            )
        moments = self.predict_moments(atoms, seeds)
        out = self.runtime.efs(atoms, moments)
        self.results["magmoms"] = moments
        self.results["energy"] = out["energy"]
        self.results["free_energy"] = out["energy"]
        self.results["forces"] = out["forces"]
        self.results["stress"] = full_3x3_to_voigt_6_stress(out["stress_3x3"])
