# https://github.com/amirgholami/PyHessian/blob/master/pyhessian/hessian.py

import numpy as np
import torch
import tqdm

from ..typing import DataLoader, Device, List, Module, Tensor, Tuple, Union
from .utils import (
    get_params_and_grads,
    group_add,
    group_product,
    hvp,
    normalize,
    orthonormalize,
)

__all__ = [
    "Hessian",
]


def dataloader_hvp(
    model: Module,
    dataloader: DataLoader,
    params: List[Tensor],
    criterion: Module,
    v: List[Tensor],
    device: Device,
) -> Tuple[float, List[Tensor]]:
    """
    Compute the Hessian-vector product (HVP) for full dataset.

    ### Parameters:
    - `model` (Module): The model to compute HVP for.
    - `dataloader` (DataLoader): DataLoader containing batches of data.
    - `params` (List[Tensor]): Vectorized model parameters.
    - `criterion` (Module): Loss function.
    - `v` (List[Tensor]): Vector to multiply with Hessian.
    - `device` (Device): Device to perform computation on.

    ### Returns:
    - `Tuple[float, List[Tensor]]`: Eigenvalue and Hessian-vector product.
    """
    num_data: int = 0
    THv: List[Tensor] = [
        torch.zeros(p.size()).to(device) for p in params
    ]  # accumulate Hv result

    for inputs, targets in dataloader:
        batch_size = inputs.size(0)
        num_data += float(batch_size)

        model.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward(create_graph=True)

        params, gradsH = get_params_and_grads(model)

        model.zero_grad()
        Hv = torch.autograd.grad(
            gradsH,
            params,
            grad_outputs=v,
            retain_graph=False,
            only_inputs=True,
        )
        THv = [THv1 + Hv1 * float(batch_size) + 0.0 for THv1, Hv1 in zip(THv, Hv)]

    THv = [THv1 / float(num_data) for THv1 in THv]
    eigval = group_product(THv, v).cpu().item()

    return eigval, THv


def hessian_evd(
    model: Module,
    dataloader: DataLoader,
    full_dataset: bool,
    params: List[Tensor],
    gradsH: List[Tensor],
    criterion: Module,
    device: Device,
    top_n: int = 1,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> Tuple[List[float], List[Tensor]]:
    """
    Compute the top eigenvalues and eigenvectors of the Hessian using power iteration.

    ### Parameters:
    - `model` (Module): The model to compute Hessian for.
    - `dataloader` (DataLoader): DataLoader containing batches of data.
    - `full_dataset` (bool): If `True`, computes on full dataset using `dataloader_hvp`.
    - `params` (List[Tensor]): Vectorized model parameters.
    - `gradsH` (List[Tensor]): Gradients with respect to parameters.
    - `criterion` (Module): Loss function.
    - `device` (Device): Device to perform computation on.
    - `top_n` (int): Number of top eigenvalues/eigenvectors to compute.
    - `max_iter` (int): Maximum number of iterations for power iteration.
    - `tol` (float): Convergence tolerance.

    ### Returns:
    - `Tuple[List[float], List[Tensor]]`: Top eigenvalues and corresponding eigenvectors.
    """
    assert top_n >= 1

    eigvals: List[float] = []
    eigvecs: List[Tensor] = []

    computed_dim = 0

    while computed_dim < top_n:
        eigval = None
        v = [torch.randn(p.size()).to(device) for p in params]  # generate random vector
        v = normalize(v)  # normalize the vector

        for _ in range(max_iter):
            v = orthonormalize(v, eigvecs)
            model.zero_grad()

            if full_dataset:
                _eigval, Hv = dataloader_hvp(
                    model, dataloader, params, criterion, v, device
                )
            else:
                Hv = hvp(gradsH, params, v)
                _eigval = group_product(Hv, v).cpu().item()

            v = normalize(Hv)

            if eigval == None:
                eigval = _eigval
            else:
                if abs(eigval - _eigval) / (abs(eigval) + 1e-6) < tol:
                    break
                else:
                    eigval = _eigval

        eigvals.append(eigval)
        eigvecs.append(v)
        computed_dim += 1

    return eigvals, eigvecs


def hessian_trace(
    model: Module,
    dataloader: DataLoader,
    full_dataset: bool,
    params: List[Tensor],
    gradsH: List[Tensor],
    criterion: Module,
    device: Device,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> List[float]:
    """
    Compute the trace of the Hessian using Hutchinson's method.

    ### Parameters:
    - `model` (Module): The model to compute Hessian for.
    - `dataloader` (DataLoader): DataLoader containing batches of data.
    - `full_dataset` (bool): If `True`, computes on full dataset using `dataloader_hvp`.
    - `params` (List[Tensor]): Vectorized model parameters.
    - `gradsH` (List[Tensor]): Gradients with respect to parameters.
    - `criterion` (Module): Loss function.
    - `device` (Device): Device to perform computation on.
    - `max_iter` (int): Maximum number of iterations for trace estimation.
    - `tol` (float): Convergence tolerance.

    ### Returns:
    - `List[float]`: Trace of the Hessian matrix.
    """
    trace_vHv = []
    trace = 0.0

    for _ in range(max_iter):
        model.zero_grad()
        v = [torch.randint_like(p, high=2, device=device) for p in params]

        # generate Rademacher random variables
        for vi in v:
            vi[vi == 0] = -1

        if full_dataset:
            _, Hv = dataloader_hvp(model, dataloader, params, criterion, v, device)
        else:
            Hv = hvp(gradsH, params, v)

        trace_vHv.append(group_product(Hv, v).cpu().item())

        if abs(np.mean(trace_vHv) - trace) / (abs(trace) + 1e-6) < tol:
            return trace_vHv
        else:
            trace = np.mean(trace_vHv)

    return trace_vHv


def hessian_eigval_density(
    model: Module,
    dataloader: DataLoader,
    full_dataset: bool,
    params: List[Tensor],
    gradsH: List[Tensor],
    criterion: Module,
    device: Device,
    num_runs: int = 1,
    trace_iter: int = 100,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Compute the eigenvalue density using the stochastic Lanczos quadrature (SLQ) algorithm.

    ### Parameters:
    - `model` (Module): The model to compute Hessian for.
    - `dataloader` (DataLoader): DataLoader containing batches of data.
    - `full_dataset` (bool): If `True`, computes on full dataset using `dataloader_hvp`.
    - `params` (List[Tensor]): Vectorized model parameters.
    - `gradsH` (List[Tensor]): Gradients with respect to parameters.
    - `criterion` (Module): Loss function.
    - `device` (Device): Device to perform computation on.
    - `num_runs` (int): Number of runs for SLQ.
    - `trace_iter` (int): Number of iterations for trace estimation.

    ### Returns:
    - `Tuple[List[List[float]], List[List[float]]]`: Eigenvalues and weights for each run.
    """
    eigval_list_full = []
    weight_list_full = []

    for _ in range(num_runs):
        v = [torch.randint_like(p, high=2, device=device) for p in params]
        # generate Rademacher random variables
        for vi in v:
            vi[vi == 0] = -1
        v = normalize(v)

        # standard lanczos algorithm initlization
        v_list = [v]
        w_list = []
        alpha_list = []
        beta_list = []
        ############### Lanczos
        for i in range(trace_iter):
            model.zero_grad()
            w_prime = [torch.zeros(p.size()).to(device) for p in params]
            if i == 0:
                if full_dataset:
                    _, w_prime = dataloader_hvp(
                        model, dataloader, params, criterion, v, device
                    )
                else:
                    w_prime = hvp(gradsH, params, v)
                alpha = group_product(w_prime, v)
                alpha_list.append(alpha.cpu().item())
                w = group_add(w_prime, v, -alpha)
                w_list.append(w)
            else:
                beta = torch.sqrt(group_product(w, w))
                beta_list.append(beta.cpu().item())
                if beta_list[-1] != 0.0:
                    # we should re-orth it
                    v = orthonormalize(w, v_list)
                    v_list.append(v)
                else:
                    # generate a new vector
                    w = [torch.randn(p.size()).to(device) for p in params]
                    v = orthonormalize(w, v_list)
                    v_list.append(v)
                if full_dataset:
                    _, w_prime = dataloader_hvp(
                        model, dataloader, params, criterion, v, device
                    )
                else:
                    w_prime = hvp(gradsH, params, v)
                alpha = group_product(w_prime, v)
                alpha_list.append(alpha.cpu().item())
                w_tmp = group_add(w_prime, v, -alpha)
                w = group_add(w_tmp, v_list[-2], -beta)

        T = torch.zeros(trace_iter, trace_iter).to(device)
        for i in range(len(alpha_list)):
            T[i, i] = alpha_list[i]
            if i < len(alpha_list) - 1:
                T[i + 1, i] = beta_list[i]
                T[i, i + 1] = beta_list[i]
        eigenvalues, eigenvectors = torch.linalg.eig(T)

        eigval_list = eigenvalues.real
        weight_list = torch.pow(eigenvectors[0, :], 2)
        eigval_list_full.append(list(eigval_list.cpu().numpy()))
        weight_list_full.append(list(weight_list.cpu().numpy()))

    return eigval_list_full, weight_list_full


class Hessian:
    """
    The class for computing:
    - the top-n eigenvalue(s)/eigenvector(s) of the neural network
    - the trace of the entire neural network
    - the estimated eigenvalue density
    """

    def __init__(
        self,
        model: Module,
        data_or_dataloader: Union[Tuple[Tensor, Tensor], DataLoader],
        criterion: Module,
        cuda: bool = True,
    ) -> None:
        """
        - `model` (Module): The model to compute Hessian for.
        - `data_or_dataloader` (DataLoader): Tuple of (inputs, targets) or DataLoader containing batches of data.
        - `criterion` (Module): Loss function.
        - `cuda` (bool): If `True`, uses CUDA for computation.
        """
        self.model = model.eval()
        self.data = data_or_dataloader
        self.criterion = criterion

        self.full_dataset = False
        if isinstance(self.data, DataLoader):
            self.full_dataset = True
        else:
            assert (
                len(self.data) == 2
                and isinstance(self.data[0], Tensor)
                and isinstance(self.data[1], Tensor)
            )

        if cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            self.inputs = self.inputs.to(self.device)
            self.targets = self.targets.to(self.device)

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_and_grads(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def evd(
        self,
        top_n: int = 1,
        max_iter: int = 100,
        tol: float = 1e-3,
    ) -> Tuple[List[float], List[Tensor]]:
        """
        Compute the top eigenvalues and eigenvectors of the Hessian using power iteration.

        ### Parameters:
        - `top_n` (int): Number of top eigenvalues/eigenvectors to compute.
        - `max_iter` (int): Maximum number of iterations for power iteration.
        - `tol` (float): Convergence tolerance.

        ### Returns:
        - `Tuple[List[float], List[Tensor]]`: Top eigenvalues and corresponding eigenvectors.
        """
        return hessian_evd(
            self.model,
            self.data,
            self.full_dataset,
            self.params,
            self.gradsH,
            self.criterion,
            self.device,
            top_n,
            max_iter,
            tol,
        )

    def trace(
        self,
        max_iter: int = 100,
        tol: float = 1e-3,
    ) -> List[float]:
        """
        Compute the trace of the Hessian using Hutchinson's method.

        ### Parameters:
        - `max_iter` (int): Maximum number of iterations for trace estimation.
        - `tol` (float): Convergence tolerance.

        ### Returns:
        - `List[float]`: Trace of the Hessian matrix.
        """
        return hessian_trace(
            self.model,
            self.data,
            self.full_dataset,
            self.params,
            self.gradsH,
            self.criterion,
            self.device,
            max_iter,
            tol,
        )

    def eigval_density(
        self,
        num_runs: int = 1,
        trace_iter: int = 100,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Compute the eigenvalue density using the stochastic Lanczos quadrature (SLQ) algorithm.

        ### Parameters:
        - `num_runs` (int): Number of runs for SLQ.
        - `trace_iter` (int): Number of iterations for trace estimation.

        ### Returns:
        - `Tuple[List[List[float]], List[List[float]]]`: Eigenvalues and weights for each run.
        """
        return hessian_eigval_density(
            self.model,
            self.data,
            self.full_dataset,
            self.params,
            self.gradsH,
            self.criterion,
            self.device,
            num_runs,
            trace_iter,
        )
