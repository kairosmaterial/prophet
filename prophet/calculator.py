from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from .graph import atomic_numbers_to_indices, dict_to_pytorch_geometric, preprocess_graph
from .model import load_model, scatter


class KairosCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model_path, use_kernel=True, use_compile=False, **kwargs):
        super().__init__(**kwargs)
        if use_kernel:
            assert torch.cuda.is_available(), "kernels require a CUDA device"
        self.model, config = load_model(Path(model_path), use_kernel=use_kernel)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.compile_state = False if use_compile and torch.cuda.is_available() else True
        self.atom_indices = atomic_numbers_to_indices(config["atomic_numbers"])
        self.cutoff = max(config["cutoffs"])
        self.config = config

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        graph = dict_to_pytorch_geometric(
            preprocess_graph(atoms, self.atom_indices, self.cutoff)
        )
        graph.n_graph = torch.zeros(graph.x.shape[0], dtype=torch.int64).to(self.device)
        graph = graph.to(self.device)
        if not self.compile_state:
            from torch.fx.experimental.proxy_tensor import make_fx

            self.model = torch.compile(
                make_fx(
                    self.model,
                    tracing_mode="symbolic",
                    _allow_non_fake_inputs=True,
                    _error_on_data_dependent_ops=True,
                )(
                    graph.x,
                    graph.positions,
                    graph.edge_attr,
                    graph.edge_index,
                    getattr(graph, "cell", None),
                    graph.n_node,
                    graph.n_edge,
                    graph.n_graph,
                )
            )
            self.compile_state = True
        energy_per_atom, forces, stress = self.model(
            graph.x,
            graph.positions,
            graph.edge_attr,
            graph.edge_index,
            getattr(graph, "cell", None),
            graph.n_node,
            graph.n_edge,
            graph.n_graph,
        )
        energy = scatter(energy_per_atom, graph.n_graph, dim=0, dim_size=graph.n_node.size(0))
        energy = np.array(energy.detach().cpu()[0])
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = np.array(forces.detach().cpu())
        self.results["stress"] = (
            full_3x3_to_voigt_6_stress(np.array(stress.detach().cpu()[0]))
            if stress is not None
            else None
        )
