import math
import random
import torch
import util
from util import MakeZeros
from util import savePlot

class CosineTaylorEstimator(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def set_parameters(**kwargs):
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
        assert P >= 2
        global CLAMP_UPDATE
        if "CLAMP_UPDATE" in kwargs:
            CLAMP_UPDATE = kwargs["CLAMP_UPDATE"]
        else:
            CLAMP_UPDATE = False

    @staticmethod
    def forward(ctx, grid_indices_here, posterior, coefficients):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        grid_indices_here = grid_indices_here * 2 * math.pi/GRID
        n_inputs, n_batch = posterior.size()
        initialized = (grid_indices_here.data * posterior).detach().sum(dim=0).data.clone()
        result = initialized.clone()

        momentum = MakeZeros(GRID)

        exponents = [P+2*i for i in range(coefficients.size()[0])]
        coefficients_list = coefficients.cpu().numpy().tolist()
        for itera in range(30):

          if OPTIMIZER_VERBOSE:
            loss = sum([lambda_Pa * (((1-SQUARED_SENSORY_SIMILARITY(result.unsqueeze(0) - grid_indices_here)).pow(Pa/2)) * posterior.detach()).sum(dim=0) for Pa, lambda_Pa in zip(exponents, coefficients_list)])
          loss_gradient = sum([lambda_Pa * (Pa/2 * (SQUARED_SENSORY_DIFFERENCE(result.unsqueeze(0) - grid_indices_here)) * ((1-SQUARED_SENSORY_SIMILARITY(result.unsqueeze(0) - grid_indices_here)).pow(Pa/2-1)) * posterior.detach()).sum(dim=0) for Pa, lambda_Pa in zip(exponents, coefficients_list)])

          loss_gradient2 = sum([lambda_Pa * (Pa/2 * (SQUARED_SENSORY_SIMILARITY(result.unsqueeze(0) - grid_indices_here)) * ((1-SQUARED_SENSORY_SIMILARITY(result.unsqueeze(0) - grid_indices_here)).pow(Pa/2-1)) * posterior.detach()).sum(dim=0) / 1 for Pa, lambda_Pa in zip(exponents, coefficients_list)])
          loss_gradient2 = loss_gradient2 + sum([lambda_Pa * (Pa/2 * (Pa/2-1) * (SQUARED_SENSORY_DIFFERENCE(result.unsqueeze(0) - grid_indices_here)).pow(2) * ((1-SQUARED_SENSORY_SIMILARITY(result.unsqueeze(0) - grid_indices_here)).pow(Pa/2-2)) * posterior.detach()).sum(dim=0) / 1 if Pa > 2 else 0 for Pa, lambda_Pa in zip(exponents, coefficients_list)])

          MASK = (loss_gradient2 > 0.001)
          updateGD = - loss_gradient /loss_gradient2.abs().clamp(min=0.0001)
          updateNewton = - loss_gradient/loss_gradient2

          if P != 3:
             update = torch.where(MASK, updateNewton, updateGD)
          else:
             update = - (1/(1+itera/10))  * loss_gradient

          result = result + update
          if OPTIMIZER_VERBOSE:
             print(itera, float(loss.mean()), "max absolute gradient after GD steps", loss_gradient.abs().max(), sum(MASK.float()), "Newton steps", "max update", update.abs().max())
          if float(loss_gradient.abs().max()) < 1e-6:
              break
        if random.random() < 0.01:
            print("Newton Iterations", itera)
        if float(loss_gradient.abs().max()) >= 1e-4:
            print("WARNING", float(loss_gradient.abs().max()), "after ", itera, " Newton steps")
            assert False
        ctx.save_for_backward(grid_indices_here, posterior, result, coefficients)
        return result * GRID / (2*math.pi)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grid_indices_here, posterior, result, coefficients = ctx.saved_tensors
        initialized = (grid_indices_here.data * posterior).detach().sum(dim=0).data

        exponents = [P+2*i for i in range(coefficients.size()[0])]
        coefficients_list = coefficients.cpu().numpy().tolist()


        n_inputs, n_batch = posterior.size()

        F_by_term = [(Pa/2 * (SQUARED_SENSORY_DIFFERENCE(result.unsqueeze(0) - grid_indices_here)) * ((1-SQUARED_SENSORY_SIMILARITY(result.unsqueeze(0) - grid_indices_here)).pow(Pa/2-1)) * posterior.detach()).sum(dim=0) for Pa, lambda_Pa in zip(exponents, coefficients_list)]

        #assert F.abs().mean() < 0.01, (F, F.abs().mean(), F.abs().max()/n_batch, n_batch, ((SQUARED_SENSORY_DIFFERENCE(result.unsqueeze(0) - grid_indices_here)) * posterior.detach()).sum(dim=0) / n_batch)

        dF_posterior = sum([lambda_Pa * Pa/2 * (SQUARED_SENSORY_DIFFERENCE(result.unsqueeze(0) - grid_indices_here)) * ((1-SQUARED_SENSORY_SIMILARITY(result.unsqueeze(0) - grid_indices_here)).pow(Pa/2-1)) for Pa, lambda_Pa in zip(exponents, coefficients_list)])
        dF_result = 0
        dF_result = dF_result + sum([lambda_Pa * (Pa/2 * (SQUARED_SENSORY_SIMILARITY(result.unsqueeze(0) - grid_indices_here)) * ((1-SQUARED_SENSORY_SIMILARITY(result.unsqueeze(0) - grid_indices_here)).pow(Pa/2-1)) * posterior.detach()).sum(dim=0) for Pa, lambda_Pa in zip(exponents, coefficients_list)])
        dF_result = dF_result + sum([lambda_Pa * (Pa/2 * (Pa/2-1) * (SQUARED_SENSORY_DIFFERENCE(result.unsqueeze(0) - grid_indices_here)).pow(2) * ((1-SQUARED_SENSORY_SIMILARITY(result.unsqueeze(0) - grid_indices_here)).pow(Pa/2-2)) * posterior.detach()).sum(dim=0) if Pa > 2 else 0  for Pa, lambda_Pa in zip(exponents, coefficients_list)])
        if OPTIMIZER_VERBOSE:
           if torch.isnan(result).any():
                print(result.size(), dF_result.size())
                assert False, result
           if torch.isnan(posterior).any():
                print(result.size(), dF_result.size())
                assert False, posterior
           if torch.isnan(dF_result).any():
                print(result[torch.isnan(dF_result)])
                print((SQUARED_STIMULUS_DIFFERENCE(result.unsqueeze(0) - grid_indices_here)).pow(2))
                print(((1+1e-10-SQUARED_STIMULUS_SIMILARITY(result.unsqueeze(0) - grid_indices_here)).pow(P/2-2)))
                print(result.size(), dF_result.size(), posterior.size())
                assert False

        dF_coefficients = torch.stack(F_by_term, dim=0) #[4,180]

        gradient_for_coefficients = - (1/dF_result).unsqueeze(0) * dF_coefficients # [4,180] dTheta/dLambda
        gradient_for_coefficients = (grad_output.unsqueeze(0) * gradient_for_coefficients).sum(dim=1)

        gradient = - ((1 / dF_result).unsqueeze(0) * dF_posterior).detach().data
        if OPTIMIZER_VERBOSE:
           if dF_result.abs().min() < .0001:
                assert False, dF_result.abs().min()
           if torch.isnan(dF_result).any():
                assert False, dF_result
        if OPTIMIZER_VERBOSE:
           if torch.isnan(dF_posterior).any():
                assert False, dF_posterior

        if OPTIMIZER_VERBOSE:
           if torch.isnan(grad_output).any():
                assert False, grad_output
        gradient = grad_output.unsqueeze(0) * gradient
        if OPTIMIZER_VERBOSE:
           if torch.isnan(gradient).any():
                assert False, gradient
        return None, gradient * GRID / (2*math.pi), gradient_for_coefficients
