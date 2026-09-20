from typing import Optional

import torch
import torch.nn as nn
from e3nn import o3


class RMSLayerNorm(torch.nn.Module):
    irreps: o3.Irreps
    eps: float
    affine: bool
    centering: bool
    std_balance_degrees: bool

    affine_weight: Optional[list[torch.Tensor]]
    affine_bias: Optional[list[torch.Tensor]]

    def __init__(
        self,
        irreps: o3.Irreps,
        eps: float = 1e-12,
        affine: bool = True,
        centering: bool = True,
        std_balance_degrees: bool = True,
    ):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees

        if self.affine:
            self.affine_weight = nn.ParameterList(
                [nn.Parameter(torch.ones(irr.mul)) for irr in self.irreps]
            )

            if self.centering:
                self.affine_bias = nn.ParameterList(
                    [
                        nn.Parameter(torch.zeros(irr.mul)) if irr.ir.l == 0 else None
                        for irr in self.irreps
                    ]
                )
            else:
                self.affine_bias = None
        else:
            self.affine_weight = None
            self.affine_bias = None

        input_slices = []
        for input_slice in irreps.slices():
            input_slices.append(input_slice)
        self.input_slices = input_slices

    def forward(self, node_input: torch.Tensor) -> torch.Tensor:
        input_chunks = []
        norms = []
        for i, (irr, input_slice) in enumerate(zip(self.irreps, self.input_slices)):
            x = node_input[:, input_slice].reshape(-1, irr.mul, irr.ir.dim)
            if self.centering and irr.ir.l == 0:
                x = x - x.mean(dim=-2, keepdim=True)
            input_chunks.append(x)

            if self.std_balance_degrees:
                weight = 1 / (irr.ir.dim * len(self.irreps))

                norm = (x**2 * weight).sum(axis=-2, keepdims=True)

                norm = norm.mean(axis=-1, keepdims=True)
                norms.append(norm)
            else:
                raise NotImplementedError("not implemented")


        norm = torch.cat(norms, dim=-1).sum(dim=-1, keepdim=True)
        norm = torch.pow(norm + self.eps, -0.5)

        out_chunks = []
        for i, (irr, x) in enumerate(zip(self.irreps, input_chunks)):
            if self.affine_weight is not None:
                x = x * self.affine_weight[i][:, None]

            out = x * norm

            if self.affine_bias is not None and irr.ir.l == 0:
                out = out + self.affine_bias[i][:, None]

            out = out.reshape(-1, irr.mul * irr.ir.dim)
            out_chunks.append(out)

        return torch.cat(out_chunks, dim=-1)
