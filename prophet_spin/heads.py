import torch
import torch.nn as nn

FEAT_DIM = 256


def features(model, batch, m0):
    species, edge_index, edge_attr, batch_idx, _ = model._common_inputs(batch)
    with torch.no_grad():
        _, last_irreps = model.model.forward_features(
            species, batch.pos, edge_attr, edge_index, batch.cell, batch_idx, magmoms=m0
        )
    return last_irreps[:, :FEAT_DIM]


def features_graph(model, batch, m0):
    species, edge_index, edge_attr, batch_idx, _ = model._common_inputs(batch)
    with torch.no_grad():
        _, last_irreps = model.model.forward_features(
            species, batch.pos, edge_attr, edge_index, batch.cell, batch_idx, magmoms=m0
        )
        r = batch.pos[edge_index[0]] - batch.pos[edge_index[1]]
        if batch.cell is not None:
            eb = torch.index_select(batch_idx, 0, edge_index[0])
            r = torch.baddbmm(
                r.view(-1, 1, 3), edge_attr.view(-1, 1, 3),
                torch.index_select(batch.cell, 0, eb),
            ).view(-1, 3)
    return last_irreps[:, :FEAT_DIM], edge_index, r.norm(dim=-1)


def _gauss(x, centers, width):
    return torch.exp(-((x.unsqueeze(-1) - centers) / width) ** 2)


class VectorMagmomHead(nn.Module):
    def __init__(self, feat_dim: int = 256, num_basis: int = 16, m2_max: float = 64.0,
                 p_max: float = 64.0, hidden: int = 128, r_basis: int = 16,
                 r_max: float = 6.0):
        super().__init__()
        self.num_basis = num_basis
        self.r_max = r_max
        self.register_buffer("m2_centers", torch.linspace(0.0, m2_max, num_basis))
        self.register_buffer("p_centers", torch.linspace(-p_max, p_max, num_basis))
        self.register_buffer("r_centers", torch.linspace(0.0, r_max, r_basis))
        self.m2_width = m2_max / (num_basis - 1) * 1.5
        self.p_width = 2 * p_max / (num_basis - 1) * 1.5
        self.r_width = r_max / (r_basis - 1) * 1.5

        self.g_self = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, num_basis + 1),
        )
        nbr_in = 2 * feat_dim + r_basis + 2 * num_basis
        self.g_nbr = nn.Sequential(
            nn.Linear(nbr_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        nn.init.zeros_(self.g_self[-1].weight); nn.init.zeros_(self.g_self[-1].bias)
        nn.init.zeros_(self.g_nbr[-1].weight); nn.init.zeros_(self.g_nbr[-1].bias)

    def forward(self, s0: torch.Tensor, h: torch.Tensor,
                edge_index: torch.Tensor, edge_len: torch.Tensor) -> torch.Tensor:
        m2 = (s0 ** 2).sum(-1)
        phi = _gauss(m2, self.m2_centers, self.m2_width)
        g = self.g_self(h)
        c_self = 1.0 + (phi * g[:, : self.num_basis]).sum(-1) + g[:, self.num_basis]
        out = c_self.unsqueeze(-1) * s0

        recv, send = edge_index[0], edge_index[1]
        p = (s0[recv] * s0[send]).sum(-1)
        x = edge_len.clamp(max=self.r_max) / self.r_max
        env = (1.0 - x ** 2) ** 2
        feat = torch.cat([h[recv], h[send],
                          _gauss(edge_len, self.r_centers, self.r_width),
                          _gauss(p, self.p_centers, self.p_width),
                          _gauss(m2[send], self.m2_centers, self.m2_width)], dim=-1)
        c_nbr = self.g_nbr(feat).squeeze(-1) * env
        return out.index_add(0, recv, c_nbr.unsqueeze(-1) * s0[send])


class MagmomHead(nn.Module):
    def __init__(self, feat_dim: int = 256, num_basis: int = 16, m2_max: float = 64.0,
                 hidden: int = 128):
        super().__init__()
        self.num_basis = num_basis
        self.register_buffer("m2_centers", torch.linspace(0.0, m2_max, num_basis))
        self.m2_width = m2_max / (num_basis - 1) * 1.5
        self.g = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, num_basis + 1),
        )

    def forward(self, m0: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(-(((m0 ** 2).unsqueeze(-1) - self.m2_centers) / self.m2_width) ** 2)
        odd = m0.unsqueeze(-1) * phi
        g = self.g(h)
        return (odd * g[:, : self.num_basis]).sum(-1) + m0 * g[:, self.num_basis]


class MagmomHeadX(nn.Module):
    def __init__(self, feat_dim: int = 256, num_basis: int = 16, m2_max: float = 64.0,
                 hidden: int = 128, e_basis: int = 12, e_max: float = 1.0):
        super().__init__()
        self.num_basis = num_basis
        self.register_buffer("m2_centers", torch.linspace(0.0, m2_max, num_basis))
        self.m2_width = m2_max / (num_basis - 1) * 1.5
        self.register_buffer("e_centers", torch.linspace(-e_max, e_max, e_basis))
        self.e_width = 2 * e_max / (e_basis - 1) * 1.5
        self.g = nn.Sequential(
            nn.Linear(feat_dim + e_basis, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, num_basis + 1),
        )

    def forward(self, m0: torch.Tensor, h: torch.Tensor, e_site: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(-(((m0 ** 2).unsqueeze(-1) - self.m2_centers) / self.m2_width) ** 2)
        odd = m0.unsqueeze(-1) * phi
        eb = torch.exp(-(((e_site).unsqueeze(-1) - self.e_centers) / self.e_width) ** 2)
        g = self.g(torch.cat([h, eb], dim=-1))
        return (odd * g[:, : self.num_basis]).sum(-1) + m0 * g[:, self.num_basis]


def features_ex(model, batch, m0):
    cap = {}
    handle = model.model.exchange.register_forward_hook(
        lambda _m, _a, out: cap.__setitem__("e", out.detach())
    )
    try:
        h = features(model, batch, m0)
    finally:
        handle.remove()
    return h, cap["e"]
