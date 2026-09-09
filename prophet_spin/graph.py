import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list


def neighbors_single(pos, cell, pbc, cutoff):
    i, j, shifts = primitive_neighbor_list(
        "ijS", pbc=list(pbc), cell=cell, positions=pos, cutoff=cutoff,
        self_interaction=False,
    )
    return i, j, shifts


def sample_from_atoms(atoms, magmoms=None):
    out = {
        "pos": torch.as_tensor(atoms.get_positions(), dtype=torch.float32),
        "atomic_numbers": torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long),
        "cell": torch.as_tensor(np.array(atoms.get_cell()), dtype=torch.float32),
        "pbc": torch.as_tensor(np.array(atoms.get_pbc()), dtype=torch.bool),
    }
    if magmoms is not None:
        mm = np.asarray(magmoms, dtype=float)
        if mm.shape not in ((len(atoms),), (len(atoms), 3)):
            raise ValueError(
                f"magmoms must be (N,) collinear or (N, 3) vector; got {mm.shape}"
            )
        out["magmoms"] = torch.as_tensor(mm, dtype=torch.float32)
    return out


class Collator:
    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def __call__(self, graphs):
        n_graphs = len(graphs)
        natoms = torch.tensor([g["pos"].shape[0] for g in graphs], dtype=torch.long)
        batch_idx = torch.repeat_interleave(torch.arange(n_graphs), natoms)
        offsets = [0] + list(natoms[:-1].cumsum(0).tolist())
        all_i, all_j, all_shifts = [], [], []
        for b, g in enumerate(graphs):
            i, j, shifts = neighbors_single(
                g["pos"].numpy(), g["cell"].numpy(), g["pbc"].numpy(), self.cutoff
            )
            all_i.append(torch.as_tensor(i + offsets[b], dtype=torch.long))
            all_j.append(torch.as_tensor(j + offsets[b], dtype=torch.long))
            all_shifts.append(torch.as_tensor(shifts, dtype=torch.float32))
        out = {
            "pos": torch.cat([g["pos"] for g in graphs], dim=0),
            "atomic_numbers": torch.cat([g["atomic_numbers"] for g in graphs], dim=0),
            "edge_index": torch.stack(
                [torch.cat(all_i, dim=0), torch.cat(all_j, dim=0)], dim=0
            ),
            "cell_offsets": torch.cat(all_shifts, dim=0),
            "cell": torch.stack([g["cell"] for g in graphs], dim=0),
            "pbc": torch.stack([g["pbc"] for g in graphs], dim=0),
            "batch": batch_idx,
            "natoms": natoms,
        }
        if "magmoms" in graphs[0]:
            out["magmoms"] = torch.cat([g["magmoms"] for g in graphs], dim=0)
        return out


class Batch:
    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, name):
        if name in self._d:
            return self._d[name]
        raise AttributeError(name)

    def __contains__(self, name):
        return name in self._d

    def to(self, device):
        self._d = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in self._d.items()
        }
        return self

    def get(self, key, default=None):
        return self._d.get(key, default)
