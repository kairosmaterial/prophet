import io
import json

import torch
import torch.nn as nn

from .backbone import SpinBackbone

ARCH_KEYS = (
    "lmax", "hidden_irreps", "n_layers", "radial_basis_size", "radial_mlp_size",
    "radial_mlp_layers", "radial_polynomial_p", "mlp_init_scale", "index_weights",
    "layer_norm", "kernel", "shift", "scale", "avg_n_neighbors", "cutoffs",
    "model_name",
)


class ProphetSpin(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        atomic_numbers = config["atomic_numbers"]
        kwargs = {k: config[k] for k in ARCH_KEYS}
        self.model = SpinBackbone(
            n_species=len(atomic_numbers),
            cutoff=float(max(config["cutoffs"])),
            magmom_conditioning=bool(config.get("magmom_conditioning", True)),
            heisenberg_mode=bool(config.get("heisenberg_mode", True)),
            exchange_r_max=config.get("exchange_r_max"),
            **kwargs,
        )
        zmap = torch.zeros(119, dtype=torch.long)
        for i, z in enumerate(atomic_numbers):
            zmap[int(z)] = i
        self.register_buffer("z2idx", zmap, persistent=True)

    def _common_inputs(self, data):
        species = self.z2idx[data.atomic_numbers.long()]
        edge_index = data.edge_index.flip(0)
        if getattr(data, "cell_offsets", None) is not None:
            edge_attr = data.cell_offsets
        else:
            edge_attr = torch.zeros(
                edge_index.shape[1], 3, device=data.pos.device, dtype=data.pos.dtype
            )
        batch_idx = data.batch
        num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 1
        return species, edge_index, edge_attr, batch_idx, num_graphs

    def _energy_per_graph(self, node_energies, batch_idx, num_graphs):
        if node_energies.dim() > 1:
            node_energies = node_energies.squeeze(-1)
        energy = torch.zeros(
            num_graphs, device=node_energies.device, dtype=node_energies.dtype
        )
        return energy.scatter_add(0, batch_idx, node_energies)

    @property
    def graph_cutoff(self) -> float:
        ex = getattr(self.model, "exchange", None)
        return max(
            float(self.model.cutoff), float(ex.r_max) if ex is not None else 0.0
        )

    @torch.no_grad()
    def forward_energy(self, data) -> torch.Tensor:
        species, edge_index, edge_attr, batch_idx, num_graphs = self._common_inputs(data)
        cell = getattr(data, "cell", None)
        mm = getattr(data, "magmoms", None)
        node_energies, _ = self.model.forward_features(
            species, data.pos, edge_attr, edge_index, cell, batch_idx, magmoms=mm
        )
        return self._energy_per_graph(node_energies, batch_idx, num_graphs)

    def forward(self, data) -> dict:
        species, edge_index, edge_attr, batch_idx, num_graphs = self._common_inputs(data)
        cell = getattr(data, "cell", None)
        natoms = getattr(data, "natoms", None)
        mm = getattr(data, "magmoms", None)
        with torch.enable_grad():
            pos_leaf = data.pos.detach().requires_grad_(True)
            node_energies, forces, stress = self.model(
                species, pos_leaf, edge_attr, edge_index, cell,
                natoms, None, batch_idx, magmoms=mm,
            )
        energy = self._energy_per_graph(node_energies, batch_idx, num_graphs)
        out = {"energy": energy, "forces": forces}
        if stress is not None:
            out["stress"] = stress
        return out


def load_spin_model(path: str) -> tuple[ProphetSpin, dict]:
    with open(path, "rb") as f:
        config = json.loads(f.readline().decode())
        state_dict = torch.load(io.BytesIO(f.read()), map_location="cpu")
    model = ProphetSpin(config)
    state_dict = {k: v for k, v in state_dict.items() if ".tp." not in k}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, config
