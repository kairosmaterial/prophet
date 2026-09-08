import matscipy.neighbours
import numpy as np
import torch
from ase.geometry import complete_cell
from torch_geometric.data import Data


def atomic_numbers_to_indices(atomic_numbers):
    return {n: i for i, n in enumerate(sorted(atomic_numbers))}


def preprocess_graph(atoms, atom_indices, cutoff):
    cell = complete_cell(atoms.cell)
    src, dst, shift = matscipy.neighbours.neighbour_list(
        "ijS", positions=atoms.positions, cell=cell, pbc=atoms.pbc, cutoff=cutoff
    )
    return {
        "n_node": np.array([len(atoms)]).astype(np.int32),
        "n_edge": np.array([len(src)]).astype(np.int32),
        "senders": dst.astype(np.int32),
        "receivers": src.astype(np.int32),
        "species": np.array([atom_indices[n] for n in atoms.get_atomic_numbers()]).astype(
            np.int32
        ),
        "positions": atoms.positions.astype(np.float32),
        "shifts": shift.astype(np.float32),
        "cell": atoms.cell.astype(np.float32) if atoms.pbc.all() else None,
    }


def dict_to_pytorch_geometric(graph_dict):
    edge_index = torch.stack(
        [torch.from_numpy(graph_dict["senders"]), torch.from_numpy(graph_dict["receivers"])],
        dim=0,
    ).long()
    cell = (
        torch.from_numpy(graph_dict["cell"])[None, :, :]
        if graph_dict["cell"] is not None
        else None
    )
    return Data(
        n_node=torch.from_numpy(graph_dict["n_node"]),
        n_edge=torch.from_numpy(graph_dict["n_edge"]),
        x=torch.from_numpy(graph_dict["species"]).long(),
        positions=torch.from_numpy(graph_dict["positions"]),
        edge_index=edge_index,
        edge_attr=torch.from_numpy(graph_dict["shifts"]),
        cell=cell,
    )
