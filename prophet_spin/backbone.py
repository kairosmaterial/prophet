from typing import Callable, List, Mapping, Optional, Sequence, Union
import math

import numpy as np
import scipy
import torch
import torch.nn as nn
from e3nn import o3
from e3nn.util.jit import compile_mode

from .norm import RMSLayerNorm


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):

    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand_as(other)


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:

    assert reduce == "sum"
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(
            size,
            dtype=(torch.float32 if src.dtype not in (torch.float32, torch.float64) else src.dtype),
            device=src.device,
        )
        return out.scatter_add_(dim, index, src.to(out.dtype))
    return out.scatter_add_(dim, index, src)


def bessel_basis(x: torch.Tensor, num_basis: int, r_max: float) -> torch.Tensor:
    prefactor = 2.0 / r_max
    bessel_weights = torch.linspace(1.0, num_basis, num_basis, device=x.device) * torch.pi
    x = x[:, None]
    return prefactor * torch.where(
        x == 0.0,
        bessel_weights / r_max,
        torch.sin(bessel_weights * x / r_max) / x,
    )


def polynomial_cutoff(x: torch.Tensor, r_max: float, p=6.0) -> torch.Tensor:
    factor = 1.0 / r_max
    x = x * factor
    out = 1.0 - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))
    return out * torch.where(x < 1.0, 1.0, 0.0)


def normalspace(n: int, device=None, dtype=None) -> torch.Tensor:

    return np.sqrt(2) * scipy.special.erfinv(np.linspace(-1.0, 1.0, n + 2)[1:-1])


@compile_mode("trace")
class normalize2mom(torch.nn.Module):
    _is_id: bool
    cst: float

    def __init__(self, f, dtype=None, device=None) -> None:
        super().__init__()
        if device is None and isinstance(f, torch.nn.Module):
            from e3nn.util._argtools import _get_device
            device = _get_device(f)
        with torch.no_grad():
            x = torch.from_numpy(normalspace(1_000_001))
            c = torch.mean(f(x) ** 2) ** 0.5
            c = c.item()
            self.f = f if np.allclose(c, 1.0) else (lambda x: f(x) / c)
        self.x = x

    def forward(self, x):
        return self.f(x)


def parity_function(phi: Callable[[float], float]) -> int:
    x = np.linspace(0.0, 10.0, 256)
    a1, a2 = phi(x), phi(-x)
    if np.max(np.abs(a1 - a2)) < 1e-5:
        return 1
    if np.max(np.abs(a1 + a2)) < 1e-5:
        return -1
    return 0


def soft_odd(x):
    return (1 - torch.exp(-(x ** 2))) * x


@compile_mode("trace")
class Activation(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        acts: List[Optional[Callable[[float], float]]] = None,
        *,
        even_act: Callable[[float], float] = torch.nn.functional.gelu,
        odd_act: Callable[[float], float] = soft_odd,
        normalize_act: bool = True,
    ) -> None:
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        if acts is None:
            acts = [{1: even_act, -1: odd_act}[ir.p] if ir.l == 0 else None for _, ir in irreps_in]
        assert len(irreps_in) == len(acts), (irreps_in, acts)
        if normalize_act:
            acts = [normalize2mom(a) if a is not None else None for a in acts]
        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError("Activation: cannot apply activation to a non-scalar input.")
                p_out = parity_function(act) if p_in == -1 else p_in
                irreps_out.append((mul, (0, p_out)))
                if p_out == 0:
                    raise ValueError("Activation: parity violated by activation.")
            else:
                irreps_out.append((mul, (l_in, p_in)))
        self.irreps_in = irreps_in
        self.irreps_out = o3.Irreps(irreps_out)
        self.acts = torch.nn.ModuleList(acts)
        self.paths = [(mul, (l, p), act) for (mul, (l, p)), act in zip(self.irreps_in, self.acts)]

    def forward(self, features, dim: int = -1):
        output = []
        index = 0
        for mul, (l, _), act in self.paths:
            ir_dim = 2 * l + 1
            if act is not None:
                x = features.narrow(dim, index, mul)
                output.append(act(x))
            else:
                x = features.narrow(dim, index, mul * ir_dim)
                output.append(x)
            index += mul * ir_dim
        if len(output) > 1:
            return torch.cat(output, dim=dim)
        if len(output) == 1:
            return output[0]
        return torch.zeros_like(features)


class Gate(torch.nn.Module):
    def __init__(
        self,
        irreps_out: Union[str, o3.Irreps],
        act: Optional[Mapping[int, torch.nn.Module]] = None,
        act_gates: Optional[Mapping[int, torch.nn.Module]] = None,
    ):
        super().__init__()
        self.irreps_out = o3.Irreps(irreps_out)
        if act is None:
            act = {1: torch.nn.SiLU(), -1: torch.nn.Tanh()}
        if act_gates is None:
            act_gates = {1: torch.nn.SiLU(), -1: torch.nn.Tanh()}
        scalars = self.irreps_out.filter(keep=["0e", "0o"])
        vectors = self.irreps_out.filter(drop=["0e", "0o"])
        scalars_extra = scalars.slice_by_mul[: scalars.dim - vectors.num_irreps]
        scalars_gates = scalars.slice_by_mul[scalars.dim - vectors.num_irreps :]
        self.scalars_gates_vectors_elemwise = o3.ElementwiseTensorProduct(scalars_gates, vectors)
        self.slice_scalars_extra = slice(0, scalars.dim - vectors.num_irreps)
        self.slice_scalars_gates = slice(scalars.dim - vectors.num_irreps, scalars.dim)
        self.slice_vectors = slice(scalars.dim, self.irreps_out.dim)
        self.scalars_extra = Activation(scalars_extra, [act[ir.p] for _, ir in scalars_extra])
        self.scalars_gates = Activation(scalars_gates, [act_gates[ir.p] for _, ir in scalars_gates])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.concatenate(
            [
                self.scalars_extra(x[:, self.slice_scalars_extra]),
                self.scalars_gates_vectors_elemwise(
                    self.scalars_gates(x[:, self.slice_scalars_gates]),
                    x[:, self.slice_vectors],
                ),
            ],
            dim=-1,
        )


class Linear(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int, use_bias: bool = False, init_scale: float = 1.0):
        super().__init__()
        self.use_bias = use_bias
        scale = math.sqrt(init_scale / in_size)
        self.weights = torch.nn.Parameter(torch.randn(in_size, out_size) * scale)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weights
        if self.use_bias and self.bias is not None:
            x = x + self.bias
        return x


class MLP(torch.nn.Module):
    def __init__(self, sizes, activation=torch.nn.SiLU, *, init_scale: float = 1.0, use_bias: bool = True):
        super().__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList(
            [
                Linear(
                    sizes[i],
                    sizes[i + 1],
                    use_bias=use_bias,
                    init_scale=init_scale if i < len(sizes) - 2 else 1.0,
                )
                for i in range(len(sizes) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class Sort(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.irreps = irreps
        slices = list(irreps.slices())
        irreps_sorted, _, inv = irreps.sort()
        self.slices_sorted = [slices[i] for i in inv]
        self.irreps_sorted = irreps_sorted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, s] for s in self.slices_sorted], dim=-1)


class NequixTorchConvolution(torch.nn.Module):
    output_irreps: o3.Irreps
    index_weights: bool = False

    radial_mlp: MLP
    linear_1: o3.Linear
    linear_2: o3.Linear
    skip: o3.Linear
    layer_norm: Optional[RMSLayerNorm]

    def __init__(
        self,
        input_irreps: o3.Irreps,
        output_irreps: o3.Irreps,
        sh_irreps: o3.Irreps,
        n_species: int,
        radial_basis_size: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_n_neighbors: float,
        index_weights: bool = False,
        layer_norm: bool = False,
        kernel: bool = False,
        scalar_dim: int = 128,
    ):
        super().__init__()
        self.output_irreps = output_irreps
        assert not index_weights, "Index weights are not supported in o3.Linear"
        self.index_weights = index_weights
        self.kernel = kernel
        self.avg_n_neighbors = avg_n_neighbors

        self.linear_1 = o3.Linear(irreps_in=input_irreps.regroup(), irreps_out=input_irreps.regroup())

        irreps_out_tp = []
        instructions_tp = []
        for i, (mul, ir_in1) in enumerate(input_irreps):
            for j, (_, ir_in2) in enumerate(sh_irreps):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in output_irreps or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_out_tp)
                        irreps_out_tp.append((mul, ir_out))
                        instructions_tp.append((i, j, k, "uvu", True))

        tp_irreps = o3.Irreps(irreps_out_tp)
        _, _, inv = tp_irreps.sort()
        instructions_tp = [instructions_tp[i] for i in inv]

        if self.kernel:
            from openequivariance import TensorProductConv, TPProblem
            tpp = TPProblem(
                input_irreps, sh_irreps, tp_irreps,
                instructions=instructions_tp,
                shared_weights=False, internal_weights=False,
            )
            self.tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False, use_opaque=False)
        else:
            self.tp = o3.TensorProduct(
                input_irreps, sh_irreps, tp_irreps,
                instructions=instructions_tp,
                shared_weights=False, internal_weights=False,
            )

        self.sort = Sort(tp_irreps)
        tp_irreps = self.sort.irreps_sorted

        self.radial_mlp = MLP(
            sizes=[scalar_dim] + [scalar_dim] * radial_mlp_layers + [tp_irreps.num_irreps],
            activation=torch.nn.SiLU,
            use_bias=False,
            init_scale=mlp_init_scale,
        )

        gate_irreps = o3.Irreps(f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e")
        output_irreps = (output_irreps + gate_irreps).regroup()

        self.linear_2 = o3.Linear(irreps_in=tp_irreps.regroup(), irreps_out=output_irreps.regroup())
        self.skip = o3.Linear(irreps_in=input_irreps.regroup(), irreps_out=output_irreps.regroup())
        self.layer_norm = (
            RMSLayerNorm(irreps=output_irreps, centering=False, std_balance_degrees=True)
            if layer_norm else None
        )
        self.gate = Gate(irreps_out=output_irreps)

    def forward(
        self,
        features: torch.Tensor,
        species: torch.Tensor,
        sh: torch.Tensor,
        radial_basis: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        edge_envelope: torch.Tensor,
    ) -> torch.Tensor:
        messages = self.linear_1(features)
        radial_message = self.radial_mlp(radial_basis)

        if self.kernel:
            messages_agg = self.sort(
                self.tp_conv(messages, sh, radial_message, receivers, senders)
            ) / np.sqrt(self.avg_n_neighbors)
        else:
            messages = self.tp(messages[senders], sh, radial_message)
            messages = messages * edge_envelope
            messages_agg = scatter(messages, receivers, dim=0, dim_size=features.size(0)) / np.sqrt(
                self.avg_n_neighbors
            )
            messages_agg = self.sort(messages_agg)

        skip = self.skip(species, features) if self.index_weights else self.skip(features)
        features = self.linear_2(messages_agg) + skip
        if self.layer_norm is not None:
            features = self.layer_norm(features)
        return self.gate(features)


class ScalarMLP(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class EQFFN(nn.Module):
    def __init__(self, irreps, epsilon=1e-8):
        super().__init__()
        self.irreps = irreps
        self.epsilon = epsilon
        self.scalar_dim = self.irreps.filter(keep=["0e"]).num_irreps
        self.num_ch = sum(ch for ch, _ in self.irreps)
        self.linear1 = o3.Linear(self.irreps, self.irreps)
        self.mlp = nn.Sequential(
            nn.Linear(self.scalar_dim + self.num_ch, self.scalar_dim * 4),
            nn.SiLU(),
            nn.Linear(self.scalar_dim * 4, self.scalar_dim + self.num_ch),
        )

    def forward(self, scalars_feat, irreps_feat):
        x = self.linear1(irreps_feat)


        t1 = 0
        slice_norm_list = []
        for ch, ir in self.irreps:
            slice_x = x[:, t1:t1 + (ch * ir.dim)].view(-1, ch, ir.dim)
            slice_norm_list.append(torch.mean(slice_x ** 2, dim=-1, keepdim=False))
            t1 += ch * ir.dim

        ctx = self.mlp(torch.cat(slice_norm_list + [scalars_feat], dim=-1))
        m1 = ctx.narrow(-1, 0, self.scalar_dim)
        m2 = ctx.narrow(-1, self.scalar_dim, self.num_ch)


        t2 = t1 = 0
        modulated_slices = []
        for ch, ir in self.irreps:
            weight = m2[:, t2:t2 + ch].unsqueeze(-1)
            slice_x = x[:, t1:t1 + (ch * ir.dim)].view(-1, ch, ir.dim)
            modulated_slices.append((slice_x * weight).view(-1, ch * ir.dim))
            t2 += ch
            t1 += ch * ir.dim

        x_modulated = torch.cat(modulated_slices, dim=-1)
        return scalars_feat + m1, irreps_feat + x_modulated


class ProphetBlock(nn.Module):
    def __init__(
        self,
        input_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        output_irreps: o3.Irreps,
        sh_irreps: o3.Irreps,
        n_species: int,
        radial_basis_size: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_n_neighbors: float,
        index_weights: bool = False,
        layer_norm: bool = False,
        kernel: bool = False,
        cutoffs: list = [],
    ):
        super().__init__()
        self.cutoffs = cutoffs
        scalar_dim = output_irreps.filter(keep=["0e"]).num_irreps
        self.scalar_dim = scalar_dim

        self.tpconv = nn.ModuleList([
            NequixTorchConvolution(
                input_irreps=input_irreps,
                output_irreps=output_irreps,
                sh_irreps=sh_irreps,
                n_species=n_species,
                radial_basis_size=radial_basis_size,
                radial_mlp_size=radial_mlp_size,
                radial_mlp_layers=radial_mlp_layers,
                mlp_init_scale=mlp_init_scale,
                avg_n_neighbors=avg_n_neighbors,
                index_weights=index_weights,
                layer_norm=layer_norm,
                kernel=kernel,
                scalar_dim=scalar_dim,
            ) for _ in cutoffs
        ])
        self.norm = nn.ModuleList([
            RMSLayerNorm(irreps=output_irreps, centering=False, std_balance_degrees=True)
            for _ in cutoffs
        ])
        self.eq_ffn = nn.ModuleList([EQFFN(output_irreps) for _ in cutoffs])
        self.mix_irreps = o3.Linear(output_irreps * len(cutoffs), output_irreps)
        self.mix_scalars = nn.Linear(scalar_dim * len(cutoffs), scalar_dim)

    def forward(
        self,
        node_scalars,
        node_irreps,
        species,
        sh,
        multi_edge_scalars,
        senders,
        receivers,
        medge_masks,
        multi_edge_envelopes,
    ):

        if node_irreps is None:
            node_irreps = node_scalars


        multi_node_irreps = []
        for i, _ in enumerate(self.cutoffs):
            edge_mask = medge_masks[i]
            multi_node_irreps.append(self.tpconv[i](
                node_irreps, species, sh[edge_mask], multi_edge_scalars[i],
                senders[edge_mask], receivers[edge_mask], multi_edge_envelopes[i],
            ))


        for i in range(len(self.cutoffs)):
            multi_node_irreps[i] = self.norm[i](multi_node_irreps[i])
        multi_irreps, multi_scalars = [], []
        for i in range(len(self.cutoffs)):
            mscalars = multi_node_irreps[i].narrow(-1, 0, self.scalar_dim)
            new_scalars, new_irreps = self.eq_ffn[i](mscalars, multi_node_irreps[i])
            multi_scalars.append(new_scalars)
            multi_irreps.append(new_irreps)

        new_node_irreps = self.mix_irreps(torch.cat(multi_irreps, dim=-1))
        if node_irreps.shape == new_node_irreps.shape:
            new_node_irreps = new_node_irreps + node_irreps
        new_node_scalars = self.mix_scalars(torch.cat(multi_scalars, dim=-1)) + node_scalars
        return new_node_scalars, new_node_irreps


def envelope_fn2(d, cutoff) -> torch.Tensor:
    p = 5.0
    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2
    d_scaled = d / cutoff
    env_val = 1 + a * d_scaled ** p + b * d_scaled ** (p + 1) + c * d_scaled ** (p + 2)
    return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start: float = -5.0, stop: float = 5.0, num_gaussians: int = 50,
                 basis_width_scalar: float = 1.0) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class MagmomGraft(torch.nn.Module):

    def __init__(self, scalar_dim: int, n_cutoffs: int, num_basis: int = 16,
                 m2_max: float = 32.0, p_max: float = 32.0, q_max: float = 64.0,
                 heisenberg_mode: bool = False):
        super().__init__()
        self.num_basis = num_basis
        self.register_buffer("m2_centers", torch.linspace(0.0, m2_max, num_basis))
        self.register_buffer("p_centers", torch.linspace(-p_max, p_max, num_basis))
        self.register_buffer("q_centers", torch.linspace(0.0, q_max, num_basis))
        self.m2_width = m2_max / (num_basis - 1) * 1.5
        self.p_width = 2 * p_max / (num_basis - 1) * 1.5
        self.q_width = q_max / (num_basis - 1) * 1.5
        def zlin():
            l = torch.nn.Linear(num_basis, scalar_dim)
            torch.nn.init.zeros_(l.weight); torch.nn.init.zeros_(l.bias)
            return l
        self.node_lin = zlin()


        self.p_lin = None if heisenberg_mode else torch.nn.ModuleList(
            [zlin() for _ in range(n_cutoffs)])
        self.q_lin = torch.nn.ModuleList([zlin() for _ in range(n_cutoffs)])

    @staticmethod
    def _basis(x, centers, width):
        return torch.exp(-((x.unsqueeze(-1) - centers) / width) ** 2)

    def node(self, magmoms):
        m2 = (magmoms ** 2).sum(-1) if magmoms.dim() == 2 else magmoms ** 2
        return self.node_lin(self._basis(m2, self.m2_centers, self.m2_width))

    def edge(self, i, magmoms, senders, receivers):
        if magmoms.dim() == 2:
            p = (magmoms[senders] * magmoms[receivers]).sum(-1)
            q = (magmoms[senders] ** 2).sum(-1) + (magmoms[receivers] ** 2).sum(-1)
        else:
            p = magmoms[senders] * magmoms[receivers]
            q = magmoms[senders] ** 2 + magmoms[receivers] ** 2
        out = self.q_lin[i](self._basis(q, self.q_centers, self.q_width))
        if self.p_lin is not None:
            out = out + self.p_lin[i](self._basis(p, self.p_centers, self.p_width))
        return out


class ExchangeReadout(torch.nn.Module):

    def __init__(self, scalar_dim: int, num_basis: int = 16, q_max: float = 64.0,
                 r_max: float = 6.0, r_basis: int = 16, hidden: int = 128):
        super().__init__()
        self.r_max = r_max


        self.register_buffer("q_centers", torch.linspace(0.0, q_max, num_basis),
                             persistent=False)
        self.register_buffer("r_centers", torch.linspace(0.0, r_max, r_basis),
                             persistent=False)
        self.q_width = q_max / (num_basis - 1) * 1.5
        self.r_width = r_max / (r_basis - 1) * 1.5
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(scalar_dim + r_basis + num_basis, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h, magmoms, edge_distance, senders, receivers, n_nodes):
        if magmoms.dim() == 2:
            p = (magmoms[senders] * magmoms[receivers]).sum(-1)
            q = (magmoms[senders] ** 2).sum(-1) + (magmoms[receivers] ** 2).sum(-1)
        else:
            p = magmoms[senders] * magmoms[receivers]
            q = magmoms[senders] ** 2 + magmoms[receivers] ** 2
        feat = torch.cat([
            h[senders] + h[receivers],
            MagmomGraft._basis(edge_distance, self.r_centers, self.r_width),
            MagmomGraft._basis(q, self.q_centers, self.q_width)], dim=-1)
        j = self.mlp(feat).squeeze(-1)
        env = polynomial_cutoff(edge_distance, self.r_max)
        return scatter(0.5 * j * p * env, receivers, dim=0, dim_size=n_nodes)


class SpinBackbone(torch.nn.Module):
    def __init__(
        self,
        n_species,
        lmax: int = 3,
        cutoff: float = 5.0,
        hidden_irreps: str = "128x0e + 128x1o + 128x2e + 128x3o",
        n_layers: int = 5,
        radial_basis_size: int = 8,
        radial_mlp_size: int = 64,
        radial_mlp_layers: int = 3,
        radial_polynomial_p: float = 2.0,
        mlp_init_scale: float = 4.0,
        index_weights: bool = False,
        shift: float = 0.0,
        scale: float = 1.0,
        avg_n_neighbors: float = 1.0,
        atom_energies: Optional[Sequence[float]] = None,
        layer_norm: bool = False,
        kernel: bool = False,
        cutoffs: list = [3.0, 6.0],
        magmom_conditioning: bool = False,
        heisenberg_mode: bool = False,
        exchange_r_max: Optional[float] = None,
        model_name: str = "prophet",
    ):
        super().__init__()
        self.model_name = model_name
        self.lmax = lmax
        self.cutoff = cutoff
        self.n_species = n_species
        self.n_layers = n_layers
        self.radial_basis_size = radial_basis_size
        self.radial_polynomial_p = radial_polynomial_p
        self.shift = shift
        self.scale = scale
        self.register_buffer(
            "atom_energies",
            torch.tensor(atom_energies) if atom_energies is not None else torch.zeros(n_species),
        )

        sh_irreps = o3.Irreps.spherical_harmonics(lmax)
        hidden_irreps = o3.Irreps(hidden_irreps)
        scalar_dim = hidden_irreps.filter(keep=["0e"]).num_irreps
        input_irreps = o3.Irreps(f"{scalar_dim}x0e")
        self.hidden_irreps = hidden_irreps

        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                ProphetBlock(
                    input_irreps=input_irreps if i == 0 else hidden_irreps,
                    hidden_irreps=hidden_irreps,
                    output_irreps=hidden_irreps if i < n_layers - 1
                                  else hidden_irreps.filter(keep=["0e"]),
                    sh_irreps=sh_irreps,
                    n_species=n_species,
                    radial_basis_size=radial_basis_size,
                    radial_mlp_size=radial_mlp_size,
                    radial_mlp_layers=radial_mlp_layers,
                    mlp_init_scale=mlp_init_scale,
                    avg_n_neighbors=avg_n_neighbors,
                    index_weights=index_weights,
                    layer_norm=layer_norm,
                    kernel=kernel,
                    cutoffs=cutoffs,
                )
            )

        self.readout = o3.Linear(
            irreps_in=hidden_irreps.filter(keep=["0e"]).regroup(), irreps_out="1x0e"
        )


        self.cutoffs = cutoffs
        self.node_z_emb_fn1 = nn.Embedding(n_species, scalar_dim)
        self.node_z_emb_fn2 = nn.Embedding(n_species, scalar_dim)
        self.rbf_proj1 = nn.ModuleList([nn.Linear(self.radial_basis_size, scalar_dim) for _ in self.cutoffs])
        self.rbf_proj2 = nn.ModuleList([nn.Linear(self.radial_basis_size, scalar_dim) for _ in self.cutoffs])
        self.mc_proj = ScalarMLP(scalar_dim * len(self.cutoffs), scalar_dim, scalar_dim)
        self.update_mlp = ScalarMLP(scalar_dim * 2, scalar_dim, scalar_dim)


        self.magmom_graft = (MagmomGraft(scalar_dim, len(self.cutoffs),
                                         heisenberg_mode=heisenberg_mode)
                             if magmom_conditioning else None)


        self.exchange = (ExchangeReadout(
                            scalar_dim,
                            r_max=float(exchange_r_max) if exchange_r_max
                            else float(max(cutoffs)))
                         if (magmom_conditioning and heisenberg_mode) else None)

        if self.model_name == "prophet-06":
            self.edge_radial_fn = torch.nn.ModuleList([
                GaussianSmearing(0.0, c, self.radial_basis_size, 2.0) for c in self.cutoffs])

    def node_energies(
        self,
        displacements: torch.Tensor,
        species: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        return_irreps: bool = False,
        magmoms: Optional[torch.Tensor] = None,
    ):
        node_z_feat1 = self.node_z_emb_fn1(species)
        node_z_feat2 = self.node_z_emb_fn2(species)

        edge_distance_full = torch.linalg.norm(displacements, ord=2, dim=-1)
        senders_full, receivers_full = senders, receivers
        if self.exchange is not None and self.exchange.r_max > self.cutoff:


            tm = edge_distance_full < self.cutoff
            displacements = displacements[tm]
            senders, receivers = senders[tm], receivers[tm]
            edge_distance = edge_distance_full[tm]
        else:
            edge_distance = edge_distance_full
        medge_masks = [edge_distance < c for c in self.cutoffs]


        multi_scalars = []
        for i, c in enumerate(self.cutoffs):
            m = medge_masks[i]
            senders_, receivers_ = senders[m], receivers[m]
            edge_dist = edge_distance[m]
            eb = (self.edge_radial_fn[i](edge_dist) if self.model_name == "prophet-06"
                  else bessel_basis(edge_dist, self.radial_basis_size, c))
            edge_radial = self.rbf_proj1[i](eb)
            edge_envelope = polynomial_cutoff(edge_dist, c).unsqueeze(-1)
            multi_scalars.append(scatter(
                node_z_feat1[senders_] * edge_radial * edge_envelope,
                receivers_, dim=0, dim_size=len(species),
            ))

        node_scalars = self.mc_proj(torch.cat(multi_scalars, dim=-1))
        node_scalars = self.update_mlp(torch.cat([node_z_feat2, node_scalars], dim=-1))
        if self.magmom_graft is not None and magmoms is not None:
            node_scalars = node_scalars + self.magmom_graft.node(magmoms)


        multi_edge_scalars, multi_edge_envelopes = [], []
        for i, c in enumerate(self.cutoffs):
            m = medge_masks[i]
            senders_, receivers_ = senders[m], receivers[m]
            edge_dist = edge_distance[m]
            eb2 = (self.edge_radial_fn[i](edge_dist) if self.model_name == "prophet-06"
                   else bessel_basis(edge_dist, self.radial_basis_size, c))
            edge_radial = self.rbf_proj2[i](eb2)
            edge_envelope = polynomial_cutoff(edge_dist, c).unsqueeze(-1)
            edge_scalar = (node_scalars[senders_] + node_scalars[receivers_]) * edge_radial * edge_envelope
            if self.magmom_graft is not None and magmoms is not None:
                edge_scalar = edge_scalar + self.magmom_graft.edge(i, magmoms, senders_, receivers_) * edge_envelope
            multi_edge_scalars.append(edge_scalar)
            multi_edge_envelopes.append(edge_envelope)

        sh = o3.spherical_harmonics(
            o3.Irreps.spherical_harmonics(self.lmax),
            displacements,
            normalize=True,
            normalization="component",
        )

        node_irreps = None
        last_full_irreps = None
        for i, layer in enumerate(self.layers):
            node_scalars, node_irreps = layer(
                node_scalars, node_irreps, species, sh,
                multi_edge_scalars, senders, receivers, medge_masks, multi_edge_envelopes,
            )


            if i == len(self.layers) - 2:
                last_full_irreps = node_irreps

        node_energies = self.readout(node_scalars)
        node_energies = node_energies * self.scale + self.shift
        node_energies = node_energies + self.atom_energies[species, None]
        if self.exchange is not None and magmoms is not None:


            node_energies = node_energies + self.exchange(
                node_scalars, magmoms, edge_distance_full, senders_full,
                receivers_full, len(species)).unsqueeze(-1)

        if return_irreps:
            return node_energies, last_full_irreps
        return node_energies

    def forward(
        self,
        species: torch.Tensor,
        positions: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        cell: torch.Tensor,
        n_node: torch.Tensor,
        n_edge: torch.Tensor,
        n_graph: torch.Tensor,
        magmoms: Optional[torch.Tensor] = None,
    ):
        positions.requires_grad_(True)
        if cell is not None:
            eps = torch.zeros_like(cell)
            eps.requires_grad_(True)
            eps_sym = (eps + eps.swapaxes(1, 2)) / 2
            positions_strain = positions + torch.bmm(
                positions.unsqueeze(-2), torch.index_select(eps_sym, 0, n_graph)
            ).squeeze(-2)
            cell_strain = cell + torch.bmm(cell, eps_sym)
            r = torch.index_select(positions_strain, 0, edge_index[0]) - torch.index_select(
                positions_strain, 0, edge_index[1]
            )
            edge_index_batches = torch.index_select(n_graph, 0, edge_index[0])
            r = torch.baddbmm(
                r.view(-1, 1, 3),
                edge_attr.view(-1, 1, 3),
                torch.index_select(cell_strain, 0, edge_index_batches),
            ).view(-1, 3)
            node_energies = self.node_energies(r, species, edge_index[0], edge_index[1], magmoms=magmoms)
            minus_forces, virial = torch.autograd.grad(
                outputs=[node_energies.sum()], inputs=[positions, eps],
                create_graph=self.training, materialize_grads=True,
            )
            stress = virial / torch.abs(torch.linalg.det(cell))[:, None, None]
        else:
            r = positions[edge_index[0]] - positions[edge_index[1]]
            node_energies = self.node_energies(r, species, edge_index[0], edge_index[1], magmoms=magmoms)
            minus_forces = torch.autograd.grad(
                outputs=[node_energies.sum()], inputs=[positions],
                create_graph=self.training, materialize_grads=True,
            )[0]
            stress = None
        return node_energies[:, 0], -minus_forces, stress

    def forward_features(
        self,
        species: torch.Tensor,
        positions: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        cell: Optional[torch.Tensor],
        n_graph: torch.Tensor,
        magmoms: Optional[torch.Tensor] = None,
    ):
        if self.n_layers < 2:
            raise ValueError(
                "forward_features requires n_layers >= 2 so the penultimate "
                f"layer carries full hidden_irreps; got n_layers={self.n_layers}."
            )
        if cell is not None:
            r = positions[edge_index[0]] - positions[edge_index[1]]
            edge_index_batches = torch.index_select(n_graph, 0, edge_index[0])
            r = torch.baddbmm(
                r.view(-1, 1, 3),
                edge_attr.view(-1, 1, 3),
                torch.index_select(cell, 0, edge_index_batches),
            ).view(-1, 3)
        else:
            r = positions[edge_index[0]] - positions[edge_index[1]]
        node_energies, last_irreps = self.node_energies(
            r, species, edge_index[0], edge_index[1], return_irreps=True, magmoms=magmoms,
        )
        return node_energies[:, 0], last_irreps
