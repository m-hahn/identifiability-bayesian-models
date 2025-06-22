import math
import random
import torch
import util
from util import MakeZeros, ToDevice
from util import savePlot

averageNumberOfNewtonSteps = 2

class MAPCircularEstimator(torch.autograd.Function):
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
        global GRID
        GRID = kwargs["GRID"]
        global OPTIMIZER_VERBOSE
        OPTIMIZER_VERBOSE = kwargs["OPTIMIZER_VERBOSE"]
        global FINE_GRID_SCALE_STATIC
        FINE_GRID_SCALE_STATIC = 10
        if "FINE_GRID_SCALE_STATIC" in kwargs:
           FINE_GRID_SCALE_STATIC = kwargs["FINE_GRID_SCALE_STATIC"]
        assert FINE_GRID_SCALE_STATIC == 10
        global CLAMP_RESULT
        CLAMP_RESULT = False
        global UPDATE_DECAY_FACTOR
        UPDATE_DECAY_FACTOR = 5
        if "UPDATE_DECAY_FACTOR" in kwargs:
           UPDATE_DECAY_FACTOR = kwargs["UPDATE_DECAY_FACTOR"]
        assert UPDATE_DECAY_FACTOR == 5
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
        grid_indices_here = grid_indices_here/SCALE
        n_inputs, n_batch = posterior.size()

        # Step 1: Identify the argmax on the full discretized grid
        kernelFunctionForAll = torch.nn.functional.softmax(-(grid_indices_here.view(GRID, 1, GRID) - grid_indices_here.view(1,GRID,GRID)).pow(2) / (2*(KERNEL_WIDTH**2)), dim=1)
        smoothedPosteriorForAll = (kernelFunctionForAll * posterior.view(1,GRID,GRID)).sum(dim=1)
        resultDiscrete = smoothedPosteriorForAll.argmax(dim=0)
        resultDiscrete = torch.stack([grid_indices_here[resultDiscrete[i],i] for i in range(GRID)], dim =0)
        resultDiscrete1 = resultDiscrete
        assert resultDiscrete1.max() < 720/SCALE

        # Step 2: Improve estimator by optimizing one finer grid centered around the coarse maximum.
        # A small trust region leads to a finer grid and a better starting point of Newton's.
        # But sometimes the optimum will be found on the boundary of the trust region. In this case, we increase the size of the trust region by a factor of four and try again. NOTE: This behavior only makes sense when the optimum is not at the boundary of the overall stimulus space.
        # One issue that can make the small trust region very important is when there are abrupt spikes in the discretized poosterior.
        Q = 20
        FINE_GRID_SCALE = FINE_GRID_SCALE_STATIC
        for J in range(1):
           environment = (ToDevice(torch.arange(start=-Q, end=Q+1))/(FINE_GRID_SCALE*SCALE)).view(-1, 1) + resultDiscrete.view(1, -1)
           K = environment.size()[0]
           kernelFunction = torch.nn.functional.softmax(-(environment.view(K, 1, GRID) - grid_indices_here.view(1,GRID,GRID)).pow(2) / (2*(KERNEL_WIDTH**2)), dim=1)
           smoothedPosterior = (kernelFunction * posterior.view(1,GRID,GRID)).sum(dim=1)
           smoothedPosteriorInEnvironment = smoothedPosterior.detach()

           argmaxInFineGrid = smoothedPosterior.argmax(dim=0)
           onBoundary = torch.logical_or(argmaxInFineGrid == 0, argmaxInFineGrid== K-1)
           if onBoundary.float().sum() > 0:
               print("Warning: Some points are on the boundary of the trust region in iteration ",J, "Retrying with a larger region" if J == 0 else "Accepting nonetheless", "# of points:", onBoundary.float().sum())
               FINE_GRID_SCALE = .5
           else:
               break
        resultFromIntermediateGrid = ((argmaxInFineGrid-K/2)/(FINE_GRID_SCALE*SCALE) + resultDiscrete)

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
        if CLAMP_RESULT:
          resultDiscrete = resultDiscrete.clamp(min=MIN_GRID/SCALE, max=MAX_GRID/SCALE)
        assert resultDiscrete.max() < 720/SCALE

        optimizationSequence = []
        if True:
          for i in range(40):
             # First compute P'(theta)
             innerDerivatives = -(resultDiscrete.view(1, GRID) - grid_indices_here) / (KERNEL_WIDTH**2)
             kernelFunction = torch.nn.functional.softmax(-(resultDiscrete.view(1, GRID) - grid_indices_here).pow(2) / (2*(KERNEL_WIDTH**2)), dim=0)
             if OPTIMIZER_VERBOSE:
               smoothedPosterior = (kernelFunction * posterior).sum(dim=0)

             kernelFunctionDerivativeI = innerDerivatives * kernelFunction
             # This sum is needed repeatedly
             kernelFunctionDerivativeISum = kernelFunctionDerivativeI.sum(dim=0, keepdim=True)

             kernelDerivative = kernelFunctionDerivativeI - kernelFunction * kernelFunctionDerivativeISum
             derivativeSmoothedPosterior = (kernelDerivative * posterior).sum(dim=0)

             # First derivative has been verified against Pytorch autograd 

             if True:
               smoothedPosterior = (kernelFunction * posterior).sum(dim=0)
             # Now compute P''(theta)
             kernelFunctionSecondDerivativeI = -1/(KERNEL_WIDTH**2) * kernelFunction + innerDerivatives.pow(2) * kernelFunction
             part1 = kernelFunctionSecondDerivativeI - kernelFunctionDerivativeI * kernelFunctionDerivativeISum
             kernelSecondDerivative = part1 - kernelDerivative * kernelFunctionDerivativeISum - kernelFunction * part1.sum(dim=0, keepdim=True)
             secondDerivativeSmoothedPosterior = (kernelSecondDerivative * posterior).sum(dim=0)
             # Second derivative has been verified against finite differences of the first derivatives
             # These calculations are all bottlenecks. Could perhaps reduce some of the inefficiency.

             hessian = secondDerivativeSmoothedPosterior
             if OPTIMIZER_VERBOSE and i % 10 == 0 or False:
                print(i, smoothedPosterior.mean(), derivativeSmoothedPosterior.abs().mean(), derivativeSmoothedPosterior.abs().max(), "\t", resultDiscrete[10:15]*SCALE)

             # Optimization step
             # For those batch elements where P'' < 0, we do a Newton step.
             # For the others, we do a GD step, with stepsize based 1/P'', cutting off excessively large resulting stepsizes
             # For a well-behaved problem (concave within the trust region and attains its maximum on it), only Newton steps should be required. GD steps are intended as a fallback when this is not satisfied.
             MASK = hessian>=-0.001
             updateHessian =  - derivativeSmoothedPosterior / hessian.clamp(max=-0.001)

             updateGD = derivativeSmoothedPosterior / hessian.abs().clamp(min=0.01)
             # Prevent excessive jumps, and decay the allowed learning rate.
             MAXIMAL_UPDATE_SIZE = 0.1 / (1+i/UPDATE_DECAY_FACTOR)
             update = torch.where(MASK, updateGD, updateHessian).clamp(min=-MAXIMAL_UPDATE_SIZE, max=MAXIMAL_UPDATE_SIZE)
             if random.random() < 0.002:
               print("Maximal update", update.abs().max(), update.abs().median(), (update.abs() >= 0.1).float().sum(), "Doing Non-Newton", MASK.float().sum(), hessian.abs().median())
             resultDiscrete = resultDiscrete + update
             assert resultDiscrete.max() < 720/SCALE
             optimizationSequence.append((resultDiscrete, smoothedPosterior, derivativeSmoothedPosterior, secondDerivativeSmoothedPosterior))
             if PLOT:
                axis[0].plot([SCALE*resultDiscrete[0], SCALE*resultDiscrete[0]], [Y,Z], color="red")
                axis[1].plot([SCALE*resultDiscrete[80], SCALE*resultDiscrete[80]], [Y,Z], color="red")
             if float(derivativeSmoothedPosterior.abs().max()) < 1e-5:
                 break

             # Check whether solution has left trust region
             lowerThanTrustRegion = (environment[0] > resultDiscrete)
             higherThanTrustRegion = (environment[-1] < resultDiscrete)
             if lowerThanTrustRegion.float().sum() + higherThanTrustRegion.float().sum() > 0 and random.random() < 0.005:
                 print("Warning: some batches have left the trust region.", lowerThanTrustRegion.float().sum(), higherThanTrustRegion.float().sum())
        else:
            assert False
        averageNumberOfNewtonSteps = 0.98 * averageNumberOfNewtonSteps + (1-0.98) * i
        if random.random() < 0.003:
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
            for x, y, z, u in optimizationSequence:
               print(x[0], x[worst], "P", y[worst], "dP", z[worst], "d2P", u[worst], derivativeSmoothedPosterior[worst], resultDiscrete[worst])
               axis[0].plot([SCALE*x[0].detach().cpu(), SCALE*x[0].detach().cpu()], [Y,Z], color="yellow")
               axis[1].plot([SCALE*x[worst].detach().cpu(), SCALE*x[worst].detach().cpu()], [Y,Z], color="yellow")
            axis[0].plot([SCALE*resultDiscrete1[0].detach().cpu(), SCALE*resultDiscrete1[0].detach().cpu()], [Y,Z], color="orange")
            axis[1].plot([SCALE*resultDiscrete1[worst].detach().cpu(), SCALE*resultDiscrete1[worst].detach().cpu()], [Y,Z], color="orange")
            axis[0].plot([SCALE*resultDiscrete[0].detach().cpu(), SCALE*resultDiscrete[0].detach().cpu()], [Y,Z], color="red")
            axis[1].plot([SCALE*resultDiscrete[worst].detach().cpu(), SCALE*resultDiscrete[worst].detach().cpu()], [Y,Z], color="red")
            axis[1].plot([SCALE*resultFromIntermediateGrid[worst].detach().cpu(), SCALE*resultFromIntermediateGrid[worst].detach().cpu()], [1.2*Y,1.2*Z], color="purple")
            savePlot(f"figures/DEBUG_{__file__}.pdf")
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

        kernelFunctionSecondDerivativeI = -1/(KERNEL_WIDTH**2) * kernelFunction + innerDerivatives.pow(2) * kernelFunction
        part1 = kernelFunctionSecondDerivativeI - kernelFunctionDerivativeI * kernelFunctionDerivativeISum
        kernelSecondDerivative = part1 - kernelDerivative * kernelFunctionDerivativeISum - kernelFunction * part1.sum(dim=0, keepdim=True)
        secondDerivativeSmoothedPosterior = (kernelSecondDerivative * posterior).sum(dim=0)
        hessian = secondDerivativeSmoothedPosterior
        #################

        # Now using implicit differentiation
        gradient_implicit = - kernelDerivative / hessian
        gradient = grad_output.unsqueeze(0) * gradient_implicit

        return None, gradient*SCALE
