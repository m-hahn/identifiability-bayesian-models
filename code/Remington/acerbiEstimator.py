import matplotlib.pyplot as plt
import math
import random
import torch
import util
from util import MakeZeros
from util import savePlot

class AcerbiEstimator(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def set_parameters(**kwargs):
        global SQUARED_SENSORY_SIMILARITY
        SQUARED_SENSORY_SIMILARITY = torch.cos
        global SQUARED_SENSORY_DIFFERENCE
        SQUARED_SENSORY_DIFFERENCE = torch.sin
        global P
        P = kwargs["P"]
        assert P == 2
        global GRID
        GRID = kwargs["GRID"]
        global OPTIMIZER_VERBOSE
        if "OPTIMIZER_VERBOSE" in kwargs:
           OPTIMIZER_VERBOSE = kwargs["OPTIMIZER_VERBOSE"]
        else:
           OPTIMIZER_VERBOSE = False
        assert P >= 2
        global CLAMP_UPDATE
        if "CLAMP_UPDATE" in kwargs:
            CLAMP_UPDATE = kwargs["CLAMP_UPDATE"]
        else:
            CLAMP_UPDATE = False
        global MIXTURE_OF_GAUSSIANS_SETUP
        MIXTURE_OF_GAUSSIANS_SETUP = kwargs["MIXTURE_OF_GAUSSIANS_SETUP"]
    

    @staticmethod
    def forward(ctx, grid_indices_here, posterior):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
#        print(grid_indices_here)
#        quit()
#        grid_indices_here = grid_indices_here * 2 * math.pi/GRID
        n_inputs, n_batch = posterior.size()
        initialized = (grid_indices_here.data * posterior).detach().sum(dim=0).data.clone()
        result = initialized.clone()



#        pi_l = 0.3
#        mean_1 = -0.1
#        mean_2 = 0.1
#        sigma2_1 = 0.1
#        sigma2_2 = 0.05


        pi_l = MIXTURE_OF_GAUSSIANS_SETUP["pi_l"]
        mean_1 = MIXTURE_OF_GAUSSIANS_SETUP["mean_1"]
        mean_2 = MIXTURE_OF_GAUSSIANS_SETUP["mean_2"]
        sigma2_1 = MIXTURE_OF_GAUSSIANS_SETUP["sigma2_1"]
        sigma2_2 = MIXTURE_OF_GAUSSIANS_SETUP["sigma2_2"]


        def gaussian_function(x, mean, sigma2):
           return torch.exp(- (x-mean)**2 / (2*sigma2)) / math.sqrt(2*math.pi*sigma2)
        def derivative_of_gaussian_function_wrt_x(x, mean, sigma2):
           return - (x-mean) / sigma2 * gaussian_function(x, mean, sigma2)
        def second_derivative_of_gaussian_function_wrt_x(x, mean, sigma2):
           return ((x-mean)**2 / sigma2**2 - 1/sigma2) * gaussian_function(x, mean, sigma2)

        global LOSS_FUNCTION
        def LOSS_FUNCTION(difference):
           return (pi_l / math.sqrt(2*math.pi*sigma2_1) + (1-pi_l) / math.sqrt(2*math.pi*sigma2_2)) - pi_l * gaussian_function(difference, mean_1, sigma2_1) - (1-pi_l) * gaussian_function(difference, mean_2, sigma2_2)
        def DERIVATIVE_OF_LOSS_FUNCTION(difference):
           return - pi_l * derivative_of_gaussian_function_wrt_x(difference, mean_1, sigma2_1) - (1-pi_l) * derivative_of_gaussian_function_wrt_x(difference, mean_2, sigma2_2)
        def SECOND_DERIVATIVE_OF_LOSS_FUNCTION(difference):
           return - pi_l * second_derivative_of_gaussian_function_wrt_x(difference, mean_1, sigma2_1) - (1-pi_l) * second_derivative_of_gaussian_function_wrt_x(difference, mean_2, sigma2_2)



#        print(result)
 #       print(grid_indices_here)
  #      quit()
        intermediate_steps = [result]
        for itera in range(50):
#          print(result)
         

          loss = (((LOSS_FUNCTION((result.unsqueeze(0) - grid_indices_here)))) * posterior.detach()).sum(dim=0)
          loss_gradient = ((DERIVATIVE_OF_LOSS_FUNCTION((result.unsqueeze(0) - grid_indices_here))) * posterior.detach()).sum(dim=0)
          loss_gradient2 = ((SECOND_DERIVATIVE_OF_LOSS_FUNCTION((result.unsqueeze(0) - grid_indices_here))) * posterior.detach()).sum(dim=0)

          MASK = (loss_gradient2 > 0.001)
          updateGD = - 1/(1+.1*itera) * loss_gradient.sign() #(1/(1+.1 * itera)) * loss_gradient # / loss_gradient2.abs().clamp(min=0.1)
          updateNewton = - loss_gradient/loss_gradient2
          updateNewton = updateNewton.clamp(min=-0.2, max=0.2)

          if P != 3 or True:
             update = torch.where(MASK, updateNewton, updateGD)
          else:
             update = - (1/(1+itera/10))  * loss_gradient

          result = result + update
          intermediate_steps.append(result)
          if OPTIMIZER_VERBOSE:
             print(itera, float(loss.mean()), "max absolute gradient after GD steps", loss_gradient.abs().max(), sum(MASK.float()), "Newton steps", "max update", update.abs().max())
          if float(loss_gradient.abs().max()) < 1e-6:
              break
        if random.random() < 0.01:
            print("Newton Iterations", itera)
#        quit()
        ctx.save_for_backward(grid_indices_here, posterior, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        assert False, "Not implemented."


    def plot():
         figure, axis = plt.subplots(1, 1)
         axis.plot((grid-1.5).detach(), LOSS_FUNCTION((grid-1.5).detach()))
         filename = f"figures/{__file__}_LOSS_FUNCTION_{string_description_of_mixture_of_gaussians_setup}.pdf"
         savePlot(figure, filename, show=SHOW_PLOT)
         plt.close(figure)
         
