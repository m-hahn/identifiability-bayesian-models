import math
import random
import torch
import util
from util import MakeZeros
from util import savePlot

EPSILON = 0.0001

class CosineEstimator(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def set_parameters(**kwargs):
        global SQUARED_STIMULUS_DIFFERENCE
        SQUARED_STIMULUS_DIFFERENCE = kwargs["SQUARED_STIMULUS_DIFFERENCE"]
        global SQUARED_STIMULUS_SIMILARITY
        SQUARED_STIMULUS_SIMILARITY = kwargs["SQUARED_STIMULUS_SIMILARITY"]
        global SQUARED_SENSORY_SIMILARITY
        SQUARED_SENSORY_SIMILARITY = kwargs["SQUARED_SENSORY_SIMILARITY"]
        global SQUARED_SENSORY_DIFFERENCE
        SQUARED_SENSORY_DIFFERENCE = kwargs["SQUARED_SENSORY_DIFFERENCE"]
        global P
        P = kwargs["P"]
        global GRID
        GRID = kwargs["GRID"]
        global OPTIMIZER_VERBOSE
        OPTIMIZER_VERBOSE = kwargs["OPTIMIZER_VERBOSE"]
        global CLAMP_UPDATE
        if "CLAMP_UPDATE" in kwargs:
            CLAMP_UPDATE = kwargs["CLAMP_UPDATE"]
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
        grid_indices_here = grid_indices_here*360/GRID

        n_inputs, n_batch = posterior.size()
        initialized = (grid_indices_here.data * posterior).detach().sum(dim=0).data.clone()

        result = initialized.clone()

        momentum = MakeZeros(GRID)

        for itera in range(2000):

          result_ = result + .5 * momentum

          if True:
            loss = (((1+EPSILON-SQUARED_STIMULUS_SIMILARITY(result_.unsqueeze(0) - grid_indices_here)).pow(P/2)) * posterior.detach()).sum(dim=0).sum() / n_batch

          loss_gradient = (P/2 * (SQUARED_STIMULUS_DIFFERENCE(result_.unsqueeze(0) - grid_indices_here)) * ((1+EPSILON-SQUARED_STIMULUS_SIMILARITY(result_.unsqueeze(0) - grid_indices_here)).pow(P/2-1)) * posterior.detach()).sum(dim=0) / n_batch

          if False:
             loss_gradient2 = ((SQUARED_STIMULUS_SIMILARITY(result_.unsqueeze(0) - grid_indices_here)) * posterior.detach()).sum(dim=0) / n_batch

          momentum = .5*momentum - 5000 * loss_gradient
          assert not torch.isnan(loss_gradient).any()
          result += momentum
          print(itera, float(loss), "max absolute gradient after GD steps", loss_gradient.abs().max(), "Exponent", P)
          if float(loss_gradient.abs().max()) < 1e-7:
              break

        print("Iterations", itera)
        if float(loss_gradient.abs().max()) >= 1e-7:
            print("WARNING", float(loss_gradient.abs().max()), "after ", itera, " Newton steps")

        ctx.save_for_backward(grid_indices_here, posterior, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grid_indices_here, posterior, result = ctx.saved_tensors
        initialized = (grid_indices_here.data * posterior).detach().sum(dim=0).data

        n_inputs, n_batch = posterior.size()

        F = (P/2 * (SQUARED_STIMULUS_DIFFERENCE(result.unsqueeze(0) - grid_indices_here)) * ((1+EPSILON-SQUARED_STIMULUS_SIMILARITY(result.unsqueeze(0) - grid_indices_here)).pow(P/2-1)) * posterior.detach()).sum(dim=0)

        assert F.abs().mean() < 0.01, (F, F.abs().mean(), F.abs().max()/n_batch, n_batch, ((SQUARED_STIMULUS_DIFFERENCE(result.unsqueeze(0) - grid_indices_here)) * posterior.detach()).sum(dim=0) / n_batch)

        dF_posterior = P/2 * (SQUARED_STIMULUS_DIFFERENCE(result.unsqueeze(0) - grid_indices_here)) * ((1+EPSILON-SQUARED_STIMULUS_SIMILARITY(result.unsqueeze(0) - grid_indices_here)).pow(P/2-1))
        dF_result = 0
        dF_result = dF_result + (P/2 * (SQUARED_STIMULUS_SIMILARITY(result.unsqueeze(0) - grid_indices_here)) * ((1+EPSILON-SQUARED_STIMULUS_SIMILARITY(result.unsqueeze(0) - grid_indices_here)).pow(P/2-1)) * posterior.detach()).sum(dim=0)
        if P == 2:
           dF_result = dF_result + (P/2 * (SQUARED_STIMULUS_DIFFERENCE(result.unsqueeze(0) - grid_indices_here)).pow(2) * posterior.detach()).sum(dim=0)
        else:
           dF_result = dF_result + (P/2 * (P/2-1) * (SQUARED_STIMULUS_DIFFERENCE(result.unsqueeze(0) - grid_indices_here)).pow(2) * ((1+EPSILON-SQUARED_STIMULUS_SIMILARITY(result.unsqueeze(0) - grid_indices_here)).pow(P/2-2)) * posterior.detach()).sum(dim=0)

        gradient = - ((1 / dF_result).unsqueeze(0) * dF_posterior).detach().data

        gradient = grad_output.unsqueeze(0) * gradient

        gradient = 56*gradient
        return None, gradient
