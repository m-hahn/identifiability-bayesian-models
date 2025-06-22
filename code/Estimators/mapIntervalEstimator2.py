import math
import random
import torch
import util
from util import MakeZeros
from util import savePlot, ToDevice
from util import MakeZeros, ToDevice
from util import savePlot


averageNumberOfNewtonSteps = 2

class MAPIntervalEstimator(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def set_parameters(**kwargs):
        global KERNEL_WIDTH
        KERNEL_WIDTH = kwargs["KERNEL_WIDTH"]
        global MIN_GRID
        MIN_GRID = kwargs["MIN_GRID"]
        global MAX_GRID
        MAX_GRID = kwargs["MAX_GRID"]
        global SCALE
        SCALE = kwargs["SCALE"]
        global W
        W = kwargs["W"]
        global GRID
        GRID = kwargs["GRID"]
        global OPTIMIZER_VERBOSE
        OPTIMIZER_VERBOSE = kwargs["OPTIMIZER_VERBOSE"]
        global MAX_UPDATE_FACTOR
        global MAX_UPDATE_FACTOR
        MAX_UPDATE_FACTOR = 10
        if "MAX_UPDATE_FACTOR" in kwargs:
           MAX_UPDATE_FACTOR = kwargs["MAX_UPDATE_FACTOR"]
        assert MAX_UPDATE_FACTOR == 10
        global BREAK_THRESHOLD
        BREAK_THRESHOLD = 1e-6
        if "BREAK_THRESHOLD" in kwargs:
           BREAK_THRESHOLD = kwargs["BREAK_THRESHOLD"]
        assert BREAK_THRESHOLD == 1e-6

    # For this MAP estimator implementation, there are checks in ExampleGardelle_Inference_Discretized_Likelihood_Nonparam_MotorNoise1_Lp_Scale_NewStrict_NaturalPrior_ZeroExponent4_CROSSVALI_DebugD6_Smoothed2_UnitTest_GD_Check_Unnormalized_Hess_FineGrid_Fix2.py
    @staticmethod
    def forward(ctx, grid_indices_here, posterior):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        global averageNumberOfNewtonSteps
        # Step 1: To avoid over-/underflow, rescale the indices
        assert MIN_GRID == 0 # mapIntervalEstimator.py accounts for MIN_GRID
        grid_indices_here = grid_indices_here/SCALE
        n_inputs, n_batch = posterior.size()

        # Step 1: Identify the argmax on the full discretized grid
        resultDiscrete = posterior.argmax(dim=0)
        resultDiscrete = torch.stack([grid_indices_here[resultDiscrete[i],i] for i in range(GRID)], dim =0)
        resultDiscrete1 = resultDiscrete
        resultDiscreteFromRaw = resultDiscrete

        # Step 2: Improve estimator by optimizing one finer grid centered around the coarse maximum.
        # A small trust region leads to a finer grid and a better starting point of Newton's.
        # But sometimes the optimum will be found on the boundary of the trust region. In this case, we increase the size of the trust region by a factor of four and try again. NOTE: This behavior only makes sense when the optimum is not at the boundary of the overall stimulus space.
        # One issue that can make the small trust region very important is when there are abrupt spikes in the discretized poosterior.
        Q = 20
        FINE_GRID_SCALE = 2
        for J in range(1):
           environment = (ToDevice(torch.arange(start=-Q, end=Q+1))/(FINE_GRID_SCALE*SCALE)).view(-1, 1) + resultDiscrete.view(1, -1)
           K = environment.size()[0]
           kernelFunction = torch.nn.functional.softmax(-(environment.view(K, 1, GRID) - grid_indices_here.view(1,GRID,GRID)).pow(2) / (2*(KERNEL_WIDTH**2)), dim=1)
           smoothedPosterior = (kernelFunction * posterior.view(1,GRID,GRID)).sum(dim=1)

           extraStep = -torch.exp(-W*(environment-MIN_GRID/SCALE)) - torch.exp(W*(environment-MAX_GRID/SCALE))
           smoothedPosterior = smoothedPosterior + .1 * extraStep

           smoothedPosteriorInEnvironment = smoothedPosterior.detach()

           argmaxInFineGrid = smoothedPosterior.argmax(dim=0)
           resultFromIntermediateGrid = ((argmaxInFineGrid-K/2)/(FINE_GRID_SCALE*SCALE) + resultDiscrete)
           if False:
            onBoundary = torch.logical_and(torch.logical_or(argmaxInFineGrid == 0, argmaxInFineGrid== K-1), torch.logical_and(resultFromIntermediateGrid > grid_indices_here.min(dim=0)[0], resultFromIntermediateGrid < grid_indices_here.max(dim=0)[0]))
            if onBoundary.float().sum() > 0:
               print("Warning: Some points are on the boundary of the trust region in iteration ",J, "Retrying with a larger region" if J == 0 else "Accepting nonetheless", "# of points:", onBoundary.float().sum())
               if True:
                for B in range(GRID):
                 if onBoundary[B]:
                   print(B,resultDiscrete[B], environment[0,B], environment[K-1,B], argmaxInFineGrid[B], resultFromIntermediateGrid[B], GRID/SCALE)
                   figure, axis = plt.subplots(1, 2)

                   kernelFunction = torch.nn.functional.softmax(-(grid_indices_here.view(GRID, 1, GRID) - grid_indices_here.view(1,GRID,GRID)).pow(2) / (2*(KERNEL_WIDTH**2)), dim=1)
                   smoothedPosteriorOverAll = (kernelFunction * posterior.view(1,GRID,GRID)).sum(dim=1)

                   axis[1].scatter(SCALE*grid_indices_here[:,B].detach().cpu(), posterior[:,B].detach().cpu(), color="gray")
                   axis[1].scatter(SCALE*grid_indices_here[:,B].detach().cpu(), smoothedPosteriorOverAll[:,B].detach().cpu(), color="yellow")
                   axis[1].scatter(SCALE*environment[:,B].detach().cpu(), smoothedPosteriorInEnvironment[:,B].detach().cpu())
                   Z = max(float(smoothedPosteriorInEnvironment[:,0].max()), float(smoothedPosteriorInEnvironment[:,B].max()))
                   Y = min(float(smoothedPosteriorInEnvironment[:,0].min()), float(smoothedPosteriorInEnvironment[:,B].min()))
                   axis[1].plot([SCALE*resultDiscrete1[B].detach().cpu(), SCALE*resultDiscrete1[B].detach().cpu()], [Y,Z], color="orange")
                   axis[1].plot([SCALE*resultDiscrete[B].detach().cpu(), SCALE*resultDiscrete[B].detach().cpu()], [Y,Z], color="red")
                   axis[1].plot([SCALE*resultFromIntermediateGrid[B].detach().cpu(), SCALE*resultFromIntermediateGrid[B].detach().cpu()], [Y,Z], color="purple")

                   savePlot(f"figures/DEBUG_{__file__}.pdf")
                   plt.close()
                   quit()

                quit()
               FINE_GRID_SCALE = .5
            else:
               break

        PLOT = (random.random() < .1) and False
        if PLOT:
           # Plot the smoothed posterior
           figure, axis = plt.subplots(1, 2)
           axis[0].scatter(SCALE*environment[:,0], smoothedPosterior[:,0])
           axis[1].scatter(SCALE*environment[:,80], smoothedPosterior[:,80])
           Z = max(float(smoothedPosterior[:,0].max()), float(smoothedPosterior[:,80].max()))
           Y = min(float(smoothedPosterior[:,0].min()), float(smoothedPosterior[:,80].min()))

        if PLOT:
           axis[0].plot([SCALE*resultDiscrete[0], SCALE*resultDiscrete[0]], [Y,Z], color="orange")
           axis[0].plot([SCALE*resultFromIntermediateGrid[0], SCALE*resultFromIntermediateGrid[0]], [Y,Z], color="green")
           axis[1].plot([SCALE*resultDiscrete[80], SCALE*resultDiscrete[80]], [Y,Z], color="orange")
           axis[1].plot([SCALE*resultFromIntermediateGrid[80], SCALE*resultFromIntermediateGrid[80]], [Y,Z], color="green")

        # Now use the result from the intermediate grid
        resultDiscrete = resultFromIntermediateGrid
        resultDiscrete = resultDiscrete.clamp(min=MIN_GRID/SCALE, max=MAX_GRID/SCALE)

        optimizationSequence = []
        if True:
          for i in range(40):
             # First compute P'(theta)
             innerDerivatives = -(resultDiscrete.view(1, GRID) - grid_indices_here) / (KERNEL_WIDTH**2)
             kernelFunction = torch.nn.functional.softmax(-(resultDiscrete.view(1, GRID) - grid_indices_here).pow(2) / (2*(KERNEL_WIDTH**2)), dim=0)
             if OPTIMIZER_VERBOSE:
               smoothedPosterior = (kernelFunction * posterior).sum(dim=0)
               extraStep = -torch.exp(-W*(resultDiscrete-MIN_GRID/SCALE)) - torch.exp(W*(resultDiscrete-MAX_GRID/SCALE))
               smoothedPosterior = smoothedPosterior + .1 * extraStep

             kernelFunctionDerivativeI = innerDerivatives * kernelFunction
             # This sum is needed repeatedly
             kernelFunctionDerivativeISum = kernelFunctionDerivativeI.sum(dim=0, keepdim=True)

             kernelDerivative = kernelFunctionDerivativeI - kernelFunction * kernelFunctionDerivativeISum
             derivativeSmoothedPosterior = (kernelDerivative * posterior).sum(dim=0)
             derivativeOfExtraStep = W * torch.exp(-W*(resultDiscrete-MIN_GRID/SCALE)) - W * torch.exp(W*(resultDiscrete-MAX_GRID/SCALE))
             derivativeSmoothedPosterior = derivativeSmoothedPosterior + .1 * derivativeOfExtraStep

             # First derivative has been verified against Pytorch autograd 

             if True:
               smoothedPosterior = (kernelFunction * posterior).sum(dim=0)
             # Now compute P''(theta)
             kernelFunctionSecondDerivativeI = -1/(KERNEL_WIDTH**2) * kernelFunction + innerDerivatives.pow(2) * kernelFunction
             part1 = kernelFunctionSecondDerivativeI - kernelFunctionDerivativeI * kernelFunctionDerivativeISum
             kernelSecondDerivative = part1 - kernelDerivative * kernelFunctionDerivativeISum - kernelFunction * part1.sum(dim=0, keepdim=True)
             secondDerivativeSmoothedPosterior = (kernelSecondDerivative * posterior).sum(dim=0)
             secondDerivativeOfExtraStep = - W * W * torch.exp(-W*(resultDiscrete-MIN_GRID/SCALE)) - W * W * torch.exp(W*(resultDiscrete-MAX_GRID/SCALE))
             secondDerivativeSmoothedPosterior = secondDerivativeSmoothedPosterior + .1 * secondDerivativeOfExtraStep
             # Second derivative has been verified against finite differences of the first derivatives
             # These calculations are all bottlenecks. Could perhaps reduce some of the inefficiency.

             hessian = secondDerivativeSmoothedPosterior
             if OPTIMIZER_VERBOSE and i % 10 == 0 or False:
                print(i, smoothedPosterior.mean(), derivativeSmoothedPosterior.abs().mean(), derivativeSmoothedPosterior.abs().max(), "\t", resultDiscrete[10:15]*SCALE)

             # Optimization step
             # For those batch elements where P'' < 0, we do a Newton step.
             # For the others, we do a GD step, with stepsize based 1/P'', cutting off excessively large resulting stepsizes
             # For a well-behaved problem (concave within the trust region and attains its maximum on it), only Newton steps should be required. GD steps are intended as a fallback when this is not satisfied.
             MASK = hessian>=-0.005
             updateHessian =  - derivativeSmoothedPosterior / hessian.clamp(max=-0.001)
             updateGD = derivativeSmoothedPosterior / hessian.abs().clamp(min=0.01)
             # Prevent excessive jumps, and decay the allowed learning rate.
             MAXIMAL_UPDATE_SIZE = 0.1 / (1+i/MAX_UPDATE_FACTOR)
             update = torch.where(MASK, updateGD, updateHessian).clamp(min=-MAXIMAL_UPDATE_SIZE, max=MAXIMAL_UPDATE_SIZE)
             if random.random() < 0.002:
               print("Maximal update", update.abs().max(), update.abs().median(), (update.abs() >= 0.1).float().sum(), "Doing Non-Newton", MASK.float().sum(), hessian.abs().median())
             resultDiscrete = resultDiscrete + update
             optimizationSequence.append((resultDiscrete, smoothedPosterior, derivativeSmoothedPosterior, secondDerivativeSmoothedPosterior, MASK))
             if PLOT:
                axis[0].plot([SCALE*resultDiscrete[0], SCALE*resultDiscrete[0]], [Y,Z], color="red")
                axis[1].plot([SCALE*resultDiscrete[80], SCALE*resultDiscrete[80]], [Y,Z], color="red")

             if False:
               MaskOutsideOfRegion = torch.logical_or(resultDiscrete < MIN_GRID/SCALE, resultDiscrete > MAX_GRID/SCALE)
               derivativeSmoothedPosterior = torch.where(MaskOutsideOfRegion, 0*derivativeSmoothedPosterior, derivativeSmoothedPosterior)
               resultDiscrete = resultDiscrete.clamp(min=MIN_GRID/SCALE, max=MAX_GRID/SCALE)

             if False:
             # Check whether solution has left trust region
               lowerThanTrustRegion = (environment[0] > resultDiscrete)
               higherThanTrustRegion = (environment[-1] < resultDiscrete)
               if lowerThanTrustRegion.float().sum() + higherThanTrustRegion.float().sum() > 0:
                   print("Warning: some batches have left the trust region.", lowerThanTrustRegion.float().sum(), higherThanTrustRegion.float().sum())
             if float(derivativeSmoothedPosterior.abs().max()) < BREAK_THRESHOLD:
                 break
        else:
            assert False
        averageNumberOfNewtonSteps = 0.98 * averageNumberOfNewtonSteps + (1-0.98) * i
        if random.random() < 0.0003:
           print("Number of Newton iterations", i, "average", averageNumberOfNewtonSteps)
        if i > 20:
            print("Warning: Finding MAP estimator took", i, "iterations. Maximal gradient", float(derivativeSmoothedPosterior.abs().max()))

        if float(derivativeSmoothedPosterior.abs().max()) > 1e-4:
            print("Warning: Finding MAP estimator took", i, "iterations. Maximal gradient", float(derivativeSmoothedPosterior.abs().max()))
            worst = derivativeSmoothedPosterior.abs().argmax()
            print(sorted(derivativeSmoothedPosterior.detach().cpu().numpy().tolist()))
            print(derivativeSmoothedPosterior)
            print("PROBLEM", worst, derivativeSmoothedPosterior[worst], float(derivativeSmoothedPosterior.abs().max()), derivativeSmoothedPosterior[GRID-1])
            plt.close()
            figure, axis = plt.subplots(1, 2, figsize=(15,15))
            axis[0].scatter(SCALE*grid_indices_here[:,0].detach().cpu(), posterior[:,0].detach().cpu(), color="gray")

            kernelFunction = torch.nn.functional.softmax(-(grid_indices_here.view(GRID, 1, GRID) - grid_indices_here.view(1,GRID,GRID)).pow(2) / (2*(KERNEL_WIDTH**2)), dim=1)
            smoothedPosteriorOverAll = (kernelFunction * posterior.view(1,GRID,GRID)).sum(dim=1)

            KernelTimesMinusOnePlusKernel = kernelFunction * (kernelFunction-1)
            kernelFunctionDerivative = 2 * (grid_indices_here.view(GRID, 1, GRID) - grid_indices_here.view(1,GRID,GRID)) / (2*(KERNEL_WIDTH**2)) * KernelTimesMinusOnePlusKernel
            kernelFunctionSecondDerivative = 1 / (KERNEL_WIDTH**2) * KernelTimesMinusOnePlusKernel + ((grid_indices_here.view(GRID, 1, GRID) - grid_indices_here.view(1,GRID,GRID)) / (KERNEL_WIDTH**2)) * (kernelFunctionDerivative * (kernelFunction-1) + kernelFunction*kernelFunctionDerivative)
            hessianOverAll = (kernelFunctionSecondDerivative * posterior).sum(dim=1)
            derivativeSmoothedPosteriorOverAll = (kernelFunctionDerivative * posterior).sum(dim=1)

            axis[0].scatter(SCALE*grid_indices_here[:,0].detach().cpu(), smoothedPosteriorOverAll[:,0].detach().cpu(), color="yellow")
            axis[0].scatter(SCALE*grid_indices_here[:,0].detach().cpu(), derivativeSmoothedPosteriorOverAll[:,0].detach().cpu(), color="purple")

            axis[0].scatter(SCALE*environment[:,0].detach().cpu(), smoothedPosteriorInEnvironment[:,0].detach().cpu())
            axis[1].scatter(SCALE*grid_indices_here[:,worst].detach().cpu(), posterior[:,worst].detach().cpu(), color="gray")
            axis[1].scatter(SCALE*grid_indices_here[:,worst].detach().cpu(), smoothedPosteriorOverAll[:,worst].detach().cpu(), color="yellow")
            axis[1].scatter(SCALE*grid_indices_here[:,worst].detach().cpu(), derivativeSmoothedPosteriorOverAll[:,worst].detach().cpu(), color="purple")
            axis[1].scatter(SCALE*environment[:,worst].detach().cpu(), smoothedPosteriorInEnvironment[:,worst].detach().cpu())
            Z = max(float(smoothedPosteriorInEnvironment[:,0].max()), float(smoothedPosteriorInEnvironment[:,worst].max()))
            Y = min(float(smoothedPosteriorInEnvironment[:,0].min()), float(smoothedPosteriorInEnvironment[:,worst].min()))
            print(smoothedPosteriorInEnvironment[:,worst])
            print(smoothedPosteriorInEnvironment[:,worst].max())
            print(smoothedPosteriorInEnvironment[:,worst].min())
            print(smoothedPosteriorInEnvironment[:,0].max())
            print(smoothedPosteriorInEnvironment[:,0].min())
            for x, y, z, u, mask in optimizationSequence:
               print(x[0], x[worst], "P", y[worst], "dP", z[worst], "d2P", u[worst], derivativeSmoothedPosterior[worst], resultDiscrete[worst], mask[worst])
               axis[0].plot([SCALE*x[0].detach().cpu(), SCALE*x[0].detach().cpu()], [Y,Z], color="yellow")
               axis[1].plot([SCALE*x[worst].detach().cpu(), SCALE*x[worst].detach().cpu()], [Y,Z], color="yellow")
            axis[0].plot([SCALE*resultDiscrete1[0].detach().cpu(), SCALE*resultDiscrete1[0].detach().cpu()], [Y,Z], color="orange")
            axis[1].plot([SCALE*resultDiscrete1[worst].detach().cpu(), SCALE*resultDiscrete1[worst].detach().cpu()], [Y,Z], color="orange")
            axis[0].plot([SCALE*resultDiscrete[0].detach().cpu(), SCALE*resultDiscrete[0].detach().cpu()], [Y,Z], color="red")
            axis[1].plot([SCALE*resultDiscrete[worst].detach().cpu(), SCALE*resultDiscrete[worst].detach().cpu()], [Y,Z], color="red")
            axis[0].set_ylim(Y,Z)
            axis[1].set_ylim(Y,Z)
            savePlot(f"figures/DEBUG_{__file__}.pdf")
            plt.show()
            plt.close()
            assert False

        if PLOT:
           plt.show()
           plt.close()

        result = resultDiscrete

        ctx.save_for_backward(grid_indices_here, posterior, result)
        return result.detach()*SCALE

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grid_indices_here, posterior, result = ctx.saved_tensors

        #################
        # g'_j and g''_j are computed as in during the Newton iterations of the forward pass
        innerDerivatives = -(result.view(1, GRID) - grid_indices_here) / (KERNEL_WIDTH**2)
        kernelFunction = torch.nn.functional.softmax(-(result.view(1, GRID) - grid_indices_here).pow(2) / (2*(KERNEL_WIDTH**2)), dim=0)
        kernelFunctionDerivativeI = innerDerivatives * kernelFunction
        # This sum is needed repeatedly
        kernelFunctionDerivativeISum = kernelFunctionDerivativeI.sum(dim=0, keepdim=True)
        kernelDerivative = kernelFunctionDerivativeI - kernelFunction * kernelFunctionDerivativeISum
        derivativeSmoothedPosterior = (kernelDerivative * posterior).sum(dim=0)
        derivativeOfExtraStep = W * torch.exp(-W*(result-MIN_GRID/SCALE)) - W * torch.exp(W*(result-MAX_GRID/SCALE))
        derivativeSmoothedPosterior = derivativeSmoothedPosterior + .1 * derivativeOfExtraStep

        kernelFunctionSecondDerivativeI = -1/(KERNEL_WIDTH**2) * kernelFunction + innerDerivatives.pow(2) * kernelFunction
        part1 = kernelFunctionSecondDerivativeI - kernelFunctionDerivativeI * kernelFunctionDerivativeISum
        kernelSecondDerivative = part1 - kernelDerivative * kernelFunctionDerivativeISum - kernelFunction * part1.sum(dim=0, keepdim=True)
        secondDerivativeSmoothedPosterior = (kernelSecondDerivative * posterior).sum(dim=0)
        secondDerivativeOfExtraStep = - W * W * torch.exp(-W*(result-MIN_GRID/SCALE)) - W * W * torch.exp(W*(result-MAX_GRID/SCALE))
        secondDerivativeSmoothedPosterior = secondDerivativeSmoothedPosterior + .1 * secondDerivativeOfExtraStep
        hessian = secondDerivativeSmoothedPosterior
        #################

        # Now using implicit differentiation
        gradient_implicit = - kernelDerivative / hessian
        gradient = grad_output.unsqueeze(0) * gradient_implicit

        return None, gradient*SCALE


