import math
import random
import torch
import util
from util import MakeZeros
from util import savePlot

class LPEstimator(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def set_parameters(**kwargs):
        global SCALE
        SCALE = kwargs["SCALE"]
        global P
        P = kwargs["P"]
        global OPTIMIZER_VERBOSE
        if "OPTIMIZER_VERBOSE" in kwargs:
           OPTIMIZER_VERBOSE = kwargs["OPTIMIZER_VERBOSE"]
        else:
           OPTIMIZER_VERBOSE = False
        assert P >= 2
        global CLAMP_UPDATE
        if "CLAMP_UPDATE" in kwargs:
            CLAMP_UPDATE = kwargs[CLAMP_UPDATE]
        else:
            CLAMP_UPDATE = False

    @staticmethod
    def forward(ctx, grid_indices_here, posterior):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        grid_indices_here = grid_indices_here/SCALE
        n_inputs, n_batch = posterior.size()
        initialized = (grid_indices_here.data * posterior).detach().sum(dim=0).data
        result = initialized.clone()

        for itera in range(100):

          if OPTIMIZER_VERBOSE:
            loss = (((result.unsqueeze(0) - grid_indices_here).abs().pow(P)) * posterior.detach()).sum(dim=0).sum() / n_batch
          loss_gradient = ((P * torch.sign(result.unsqueeze(0) - grid_indices_here) * (result.unsqueeze(0) - grid_indices_here).abs().pow(P-1)) * posterior.detach()).sum(dim=0) / n_batch
          loss_gradient2 = ((P * (P-1) * (result.unsqueeze(0) - grid_indices_here).abs().pow(P-2)) * posterior.detach()).sum(dim=0) / n_batch
          result -= loss_gradient / loss_gradient2
          if OPTIMIZER_VERBOSE:
             print(itera, "max absolute gradient after Newton steps", loss_gradient.abs().max())
          if float(loss_gradient.abs().max()) < 1e-8:
              break

        if float(loss_gradient.abs().max()) >= 1e-8:
            print("WARNING", float(loss_gradient.abs().max()), "after ", itera, " Newton steps")
        if P == 2 and itera > 0:
            print("WARNING", itera, "iterations")
        ctx.save_for_backward(grid_indices_here, posterior, result)

        return SCALE*result

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grid_indices_here, posterior, result = ctx.saved_tensors
        initialized = (grid_indices_here.data * posterior).detach().sum(dim=0).data

        F = ((P * torch.sign(result.unsqueeze(0) - grid_indices_here) * (result.unsqueeze(0) - grid_indices_here).abs().pow(P-1)) * posterior.detach()).sum(dim=0)

        assert F.abs().mean() < 0.001, F
        dF_posterior = ((P * torch.sign(result.unsqueeze(0) - grid_indices_here) * (result.unsqueeze(0) - grid_indices_here).abs().pow(P-1)))
        dF_result = ((P * (P-1) * (result.unsqueeze(0) - grid_indices_here).abs().pow(P-2)) * posterior.detach()).sum(dim=0)

        gradient = - ((1 / dF_result).unsqueeze(0) * dF_posterior).detach().data

        gradient = grad_output.unsqueeze(0) * gradient

        return None, gradient*SCALE
