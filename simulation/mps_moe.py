"""Inference-only dense MoE dispatch for stable MPS graph shapes.

The checkpoint's expert weights and top-k router are unchanged. This replaces
dynamic per-expert ``where``/``index_add_`` dispatch with a weighted dense sum.
"""

import types

import torch
from torch.nn import functional as F


def _dense_forward(self, x, padding):
    flat = x.reshape(-1, x.shape[-1])
    probabilities = self.router(flat).float().softmax(-1)
    weights, indices = probabilities.topk(self.k, -1)
    weights = weights / weights.sum(-1, keepdim=True)
    routing = (F.one_hot(indices, len(self.experts)).to(weights.dtype)
               * weights[..., None]).sum(-2)
    routing = routing * padding.reshape(-1, 1).to(routing.dtype)
    output = torch.zeros_like(flat)
    for expert_index, expert in enumerate(self.experts):
        output = output + (expert(flat) * routing[:, expert_index, None]).to(flat.dtype)
    # The auxiliary load-balancing loss is unused in inference.
    return output.view_as(x), x.new_zeros(())


def install_mps_dense_moe(model):
    if next(model.parameters()).device.type != "mps":
        raise ValueError("Dense MoE dispatch is intended only for MPS inference")
    if model.training:
        raise ValueError("Dense MoE dispatch must not be used for training")
    for block in model.blocks:
        block.moe.forward = types.MethodType(_dense_forward, block.moe)
