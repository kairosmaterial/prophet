import numpy as np
import torch

from .adapter import load_spin_model
from .backbone import MagmomGraft, polynomial_cutoff
from .graph import Batch, Collator, sample_from_atoms


class PairHamiltonian:
    def __init__(self, e0, pairs_i, pairs_j, exchange, r, natoms):
        self.e0 = float(e0)
        self.pairs_i, self.pairs_j, self.exchange, self.r = pairs_i, pairs_j, exchange, r
        self.natoms = int(natoms)
        tmp = [[] for _ in range(self.natoms)]
        for k, (a, b) in enumerate(zip(pairs_i, pairs_j)):
            tmp[a].append((int(b), float(exchange[k])))
            tmp[b].append((int(a), float(exchange[k])))
        self.nbr_idx = [np.array([x[0] for x in t], dtype=np.int64) for t in tmp]
        self.nbr_exchange = [np.array([x[1] for x in t]) for t in tmp]

    def energy(self, spins):
        spins = np.asarray(spins, dtype=float)
        if spins.ndim == 1:
            dots = spins[self.pairs_i] * spins[self.pairs_j]
        else:
            dots = (spins[self.pairs_i] * spins[self.pairs_j]).sum(-1)
        return self.e0 + float((self.exchange * dots).sum())

    def delta_single(self, spins, site, new_spin):
        spins = np.asarray(spins, dtype=float)
        idx, ex = self.nbr_idx[site], self.nbr_exchange[site]
        if not len(idx):
            return 0.0
        d = np.asarray(new_spin, dtype=float) - spins[site]
        if spins.ndim == 1:
            return float((ex * (d * spins[idx])).sum())
        return float((ex * (spins[idx] @ d)).sum())

    def save(self, path):
        np.savez_compressed(
            path, e0=self.e0, pairs_i=self.pairs_i, pairs_j=self.pairs_j,
            J=self.exchange, r=self.r, natoms=self.natoms,
        )

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(float(z["e0"]), z["pairs_i"], z["pairs_j"], z["J"], z["r"],
                   int(z["natoms"]))


class SpinRuntime:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.dev = torch.device(device)
        self.model, self.config = load_spin_model(model_path)
        self.model = self.model.to(self.dev)
        self.collator = Collator(cutoff=self.model.graph_cutoff)

    def _batch(self, samples):
        return Batch(self.collator(samples)).to(self.dev)

    @torch.no_grad()
    def energy(self, atoms, magmoms) -> float:
        b = self._batch([sample_from_atoms(atoms, magmoms)])
        return float(self.model.forward_energy(b).item())

    @torch.no_grad()
    def energies_batch(self, atoms, magmoms_list, batch_size: int = 64):
        out = []
        for k in range(0, len(magmoms_list), batch_size):
            chunk = magmoms_list[k:k + batch_size]
            b = self._batch([sample_from_atoms(atoms, m) for m in chunk])
            out.append(self.model.forward_energy(b).detach().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def efs(self, atoms, magmoms):
        b = self._batch([sample_from_atoms(atoms, magmoms)])
        out = self.model(b)
        s3 = out["stress"].detach().cpu().numpy().reshape(3, 3)
        voigt = np.array([s3[0, 0], s3[1, 1], s3[2, 2], s3[1, 2], s3[0, 2], s3[0, 1]])
        return {
            "energy": float(out["energy"].item()),
            "energy_per_atom": float(out["energy"].item()) / len(atoms),
            "forces": out["forces"].detach().cpu().numpy(),
            "stress_voigt": voigt,
            "stress_3x3": s3,
        }

    @torch.no_grad()
    def extract_pair_hamiltonian(self, atoms, magmoms):
        cap = {}

        def _hook(_mod, args, _out):
            cap["h"], cap["mm"], cap["dist"] = args[0], args[1], args[2]
            cap["send"], cap["recv"] = args[3], args[4]

        handle = self.model.model.exchange.register_forward_hook(_hook)
        try:
            e_ref = self.energy(atoms, magmoms)
        finally:
            handle.remove()
        ex = self.model.model.exchange
        h, mm_t, dist = cap["h"], cap["mm"], cap["dist"]
        s, r = cap["send"], cap["recv"]
        if mm_t.dim() == 2:
            q = (mm_t[s] ** 2).sum(-1) + (mm_t[r] ** 2).sum(-1)
            p_in = (mm_t[s] * mm_t[r]).sum(-1)
        else:
            q = mm_t[s] ** 2 + mm_t[r] ** 2
            p_in = mm_t[s] * mm_t[r]
        feat = torch.cat([
            h[s] + h[r],
            MagmomGraft._basis(dist, ex.r_centers, ex.r_width),
            MagmomGraft._basis(q, ex.q_centers, ex.q_width),
        ], dim=-1)
        j_dir = 0.5 * ex.mlp(feat).squeeze(-1) * polynomial_cutoff(dist, ex.r_max)
        e0 = e_ref - float((j_dir.double() * p_in.double()).sum().item())

        si, ri = s.cpu().numpy(), r.cpu().numpy()
        jv = j_dir.cpu().numpy().astype(np.float64)
        dv = dist.cpu().numpy()
        agg = {}
        m_in = np.asarray(magmoms, dtype=float)
        for a, b, jj, dd in zip(si, ri, jv, dv):
            if a == b:
                mi2 = float((m_in[a] ** 2).sum()) if m_in.ndim == 2 else float(m_in[a] ** 2)
                e0 += jj * mi2
                continue
            key = (min(a, b), max(a, b))
            j0, d0 = agg.get(key, (0.0, np.inf))
            agg[key] = (j0 + jj, min(d0, dd))
        keys = sorted(agg)
        return PairHamiltonian(
            e0=e0,
            pairs_i=np.array([k[0] for k in keys], dtype=np.int64),
            pairs_j=np.array([k[1] for k in keys], dtype=np.int64),
            exchange=np.array([agg[k][0] for k in keys]),
            r=np.array([agg[k][1] for k in keys]),
            natoms=len(atoms),
        )
