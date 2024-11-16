# https://github.com/amirgholami/PyHessian/blob/master/pyhessian/utils.py

import torch

from ..typing import List, Module, Tensor, Tuple

__all__ = [
    "group_add",
    "group_product",
    "normalize",
    "orthonormalize",
    "get_params_and_grads",
    "hvp",
]


def group_add(
    xs: List[Tensor],
    ys: List[Tensor],
    alpha: float = 1.0,
) -> List[Tensor]:
    """xs = xs + ys*alpha"""
    return [x.data.add_(y * alpha) for x, y in zip(xs, ys)]


def group_product(
    xs: List[Tensor],
    ys: List[Tensor],
) -> Tensor:
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def normalize(
    xs: List[Tensor],
) -> List[Tensor]:
    s = group_product(xs, xs)
    s = s**0.5
    s = s.cpu().item()

    return [xi / (s + 1e-6) for xi in xs]


def orthonormalize(
    xs: List[Tensor],
    ys_list: List[List[Tensor]],
) -> List[Tensor]:
    """make xs orthogonal to each vector in ys_list"""
    for ys in ys_list:
        xs = group_add(xs, ys, alpha=-group_product(xs, ys))

    return normalize(xs)


def get_params_and_grads(
    model: Module,
) -> Tuple[List[Tensor], List[Tensor]]:
    params: List[Tensor] = []
    grads: List[Tensor] = []

    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0.0 if param.grad is None else param.grad + 0.0)

    return params, grads


def hvp(
    gradsH: List[Tensor],
    params: List[Tensor],
    v: List[Tensor],
) -> List[Tensor]:
    Hv = torch.autograd.grad(
        gradsH,
        params,
        grad_outputs=v,
        retain_graph=True,
        only_inputs=True,
    )

    return Hv
