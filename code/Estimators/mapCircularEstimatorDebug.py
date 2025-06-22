import math
import random
import torch
import util
from util import MakeZeros, ToDevice, MakeFloatTensor
from util import savePlot
import matplotlib.pyplot as plt


VERIFY = False

averageNumberOfNewtonSteps = 2

PLOT= False
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
        assert KERNEL_WIDTH < 1, "For mapCircularEstimatorDebug.py, the recommended kernel width is 0.1 or similar. For larger GRID, a smaller value might also work."
        global MIN_GRID
        MIN_GRID = kwargs["MIN_GRID"]
        global MAX_GRID
        MAX_GRID = kwargs["MAX_GRID"]
        global GRID
        GRID = kwargs["GRID"]
        global SCALE
        SCALE = GRID/(2*math.pi) #kwargs["SCALE"]
        global OPTIMIZER_VERBOSE
        OPTIMIZER_VERBOSE = kwargs["OPTIMIZER_VERBOSE"]
        global FINE_GRID_SCALE_STATIC
        FINE_GRID_SCALE_STATIC = 2
        if "FINE_GRID_SCALE_STATIC" in kwargs:
           FINE_GRID_SCALE_STATIC = kwargs["FINE_GRID_SCALE_STATIC"]
        assert FINE_GRID_SCALE_STATIC == 2
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
 #       print(grid_indices_here)
        grid = MakeFloatTensor(list(range(GRID)))/SCALE
#        print(grid)
        
        grid_indices_here = None
#        quit()
#        assert ((grid[-1] - grid[0]) - 2*math.pi).abs().max() < 1e-2, ((grid[-1] - grid[0]) - 2*math.pi).abs().max()
 #       print(grid_indices_here)
#        quit()
        n_inputs, n_batch = posterior.size()
#        print("POSTERIOR", posterior, posterior.max())

        # Step 1: Identify the argmax on the full discretized grid
        kernelFunctionForAll = torch.nn.functional.softmax(torch.cos(grid.unsqueeze(0) - grid.unsqueeze(1)) / (2*(KERNEL_WIDTH**2)), dim=1) # (where we compute value, where we get basis function from)
 #       print(grid)
  #      print((grid.unsqueeze(0) - grid.unsqueeze(1)))
   #     print(kernelFunctionForAll, kernelFunctionForAll.max())
#        quit()
        smoothedPosteriorForAll = (kernelFunctionForAll.unsqueeze(2) * posterior.view(1,GRID,GRID)).sum(dim=1)
    #    print("SMOOTHED", smoothedPosteriorForAll)
        #plt.close()
        #fig, ax = plt.subplots(1, 1)
        #for i in range(0,GRID, 10):
        #    ax.plot(range(GRID), smoothedPosteriorForAll[:,i], linestyle="dotted")
        #    ax.plot(range(GRID), posterior[:,i])
        #plt.show()
        #plt.close()
        resultDiscrete = smoothedPosteriorForAll.argmax(dim=0)
        resultDiscrete = torch.stack([grid[resultDiscrete[i]] for i in range(GRID)], dim =0)
        resultDiscrete1 = resultDiscrete
        #print(resultDiscrete1)
#        assert resultDiscrete1.max() < 

#        # Step 2: Improve estimator by optimizing one finer grid centered around the coarse maximum.
#        # A small trust region leads to a finer grid and a better starting point of Newton's.
#        # But sometimes the optimum will be found on the boundary of the trust region. In this case, we increase the size of the trust region by a factor of four and try again. NOTE: This behavior only makes sense when the optimum is not at the boundary of the overall stimulus space.
#        # One issue that can make the small trust region very important is when there are abrupt spikes in the discretized poosterior.
#        Q = 20
#        FINE_GRID_SCALE = FINE_GRID_SCALE_STATIC
#        for J in range(1):
#           environment = (ToDevice(torch.arange(start=-Q, end=Q+1))/(FINE_GRID_SCALE*SCALE)).view(-1, 1) + resultDiscrete.view(1, -1)
#           K = environment.size()[0]
#           kernelFunction = torch.nn.functional.softmax(torch.cos(environment.view(K, 1, GRID) - grid_indices_here.view(1,GRID,GRID)) / (2*(KERNEL_WIDTH**2)), dim=1)
#           smoothedPosterior = (kernelFunction * posterior.view(1,GRID,GRID)).sum(dim=1)
#           smoothedPosteriorInEnvironment = smoothedPosterior.detach()
#
#           argmaxInFineGrid = smoothedPosterior.argmax(dim=0)
#           onBoundary = torch.logical_or(argmaxInFineGrid == 0, argmaxInFineGrid== K-1)
#           if onBoundary.float().sum() > 0:
#               print("Warning: Some points are on the boundary of the trust region in iteration ",J, "Retrying with a larger region" if J == 0 else "Accepting nonetheless", "# of points:", onBoundary.float().sum())
#               FINE_GRID_SCALE = .5
#           else:
#               break
#        resultFromIntermediateGrid = ((argmaxInFineGrid-K/2)/(FINE_GRID_SCALE*SCALE) + resultDiscrete)
#
#        PLOT = True #(random.random() < .1) and False
#        if PLOT:
#           # Plot the smoothed posterior
#           figure, axis = plt.subplots(1, 2)
#           axis[0].scatter(SCALE*environment[:,0], smoothedPosterior[:,0])
#           axis[1].scatter(SCALE*environment[:,80], smoothedPosterior[:,80])
#           Z = max(float(smoothedPosterior[:,0].max()), float(smoothedPosterior[:,80].max()))
#           Y = min(float(smoothedPosterior[:,0].min()), float(smoothedPosterior[:,80].min()))
#
#        if PLOT:
#           axis[0].plot([SCALE*resultDiscrete[0], SCALE*resultDiscrete[0]], [Y,Z], color="orange")
#           axis[0].plot([SCALE*resultFromIntermediateGrid[0], SCALE*resultFromIntermediateGrid[0]], [Y,Z], color="green")
#           axis[1].plot([SCALE*resultDiscrete[80], SCALE*resultDiscrete[80]], [Y,Z], color="orange")
#           axis[1].plot([SCALE*resultFromIntermediateGrid[80], SCALE*resultFromIntermediateGrid[80]], [Y,Z], color="green")
#
#        # Now use the result from the intermediate grid
#        resultDiscrete = resultFromIntermediateGrid
#        if CLAMP_RESULT:
#          resultDiscrete = resultDiscrete.clamp(min=MIN_GRID/SCALE, max=MAX_GRID/SCALE)
#        assert resultDiscrete.max() < 720/SCALE

        optimizationSequence = []
        if True:
          for i in range(40):
             # First compute P'(theta)
             innerDerivatives = -torch.sin(resultDiscrete.view(1, GRID) - grid.unsqueeze(1)) / (2*(KERNEL_WIDTH**2))
             # Kernel(\theta_j, x_i) denotes Kernel_i(\theta_j)
             kernelFunction = torch.nn.functional.softmax(torch.cos(resultDiscrete.view(1, GRID) - grid.unsqueeze(1)) / (2*(KERNEL_WIDTH**2)), dim=0)
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
  #           print("DERIVATIVE", derivativeSmoothedPosterior)
             updateGD = derivativeSmoothedPosterior / hessian.abs().clamp(min=0.01)
             # Prevent excessive jumps, and decay the allowed learning rate.
             MAXIMAL_UPDATE_SIZE = 0.1 / (1+i/UPDATE_DECAY_FACTOR)
             update = torch.where(MASK, updateGD, updateHessian).clamp(min=-MAXIMAL_UPDATE_SIZE, max=MAXIMAL_UPDATE_SIZE)
             if random.random() < 0.002:
               print("Maximal update", update.abs().max(), update.abs().median(), (update.abs() >= 0.1).float().sum(), "Doing Non-Newton", MASK.float().sum(), hessian.abs().median())
             resultDiscrete = resultDiscrete + update
 #            print("UPDATE", update)
#             print("RESULT", resultDiscrete)
#             assert resultDiscrete.max() < 720/SCALE
             optimizationSequence.append((resultDiscrete, smoothedPosterior, derivativeSmoothedPosterior, secondDerivativeSmoothedPosterior))
             #if PLOT:
             #   axis[0].plot([SCALE*resultDiscrete[0], SCALE*resultDiscrete[0]], [Y,Z], color="red")
             #   axis[1].plot([SCALE*resultDiscrete[80], SCALE*resultDiscrete[80]], [Y,Z], color="red")
             if float(derivativeSmoothedPosterior.abs().max()) < 1e-10:
                 break

             # Check whether solution has left trust region
   #          lowerThanTrustRegion = (environment[0] > resultDiscrete)
  #           higherThanTrustRegion = (environment[-1] < resultDiscrete)
 #            if lowerThanTrustRegion.float().sum() + higherThanTrustRegion.float().sum() > 0 and random.random() < 0.005:
#                 print("Warning: some batches have left the trust region.", lowerThanTrustRegion.float().sum(), higherThanTrustRegion.float().sum())
        else:
            assert False
        averageNumberOfNewtonSteps = 0.98 * averageNumberOfNewtonSteps + (1-0.98) * i
        if random.random() < 0.003:
           print("Number of Newton iterations", i, "average", averageNumberOfNewtonSteps)
        if i > 20:
            print("Warning: Finding MAP estimator took", i, "iterations. Maximal gradient", float(derivativeSmoothedPosterior.abs().max()))

#        if float(derivativeSmoothedPosterior.abs().max()) > 1e-3:
#            print("Warning: Finding MAP estimator took", i, "iterations. Maximal gradient", float(derivativeSmoothedPosterior.abs().max()))
#            worst = derivativeSmoothedPosterior.abs().argmax()
#            print(sorted(derivativeSmoothedPosterior.detach().cpu().numpy().tolist()))
#            print(derivativeSmoothedPosterior)
#            print("PROBLEM", worst, derivativeSmoothedPosterior[worst], float(derivativeSmoothedPosterior.abs().max()), derivativeSmoothedPosterior[GRID-1])
#            plt.close()
#            figure, axis = plt.subplots(1, 2, figsize=(15,15))
#            axis[0].scatter(SCALE*grid_indices_here[:,0].detach().cpu(), posterior[:,0].detach().cpu(), color="gray")
#
#            kernelFunction = torch.nn.functional.softmax(torch.cos(grid_indices_here.view(GRID, 1, GRID) - grid_indices_here.view(1,GRID,GRID)) / (2*(KERNEL_WIDTH**2)), dim=1)
#            smoothedPosteriorOverAll = (kernelFunction * posterior.view(1,GRID,GRID)).sum(dim=1)
#
#            KernelTimesMinusOnePlusKernel = kernelFunction * (kernelFunction-1)
#            kernelFunctionDerivative = 2 * (grid_indices_here.view(GRID, 1, GRID) - grid_indices_here.view(1,GRID,GRID)) / (2*(KERNEL_WIDTH**2)) * KernelTimesMinusOnePlusKernel
#            kernelFunctionSecondDerivative = 1 / (KERNEL_WIDTH**2) * KernelTimesMinusOnePlusKernel + ((grid_indices_here.view(GRID, 1, GRID) - grid_indices_here.view(1,GRID,GRID)) / (KERNEL_WIDTH**2)) * (kernelFunctionDerivative * (kernelFunction-1) + kernelFunction*kernelFunctionDerivative)
#            hessianOverAll = (kernelFunctionSecondDerivative * posterior).sum(dim=1)
#            derivativeSmoothedPosteriorOverAll = (kernelFunctionDerivative * posterior).sum(dim=1)
#
#            axis[0].scatter(SCALE*grid_indices_here[:,0].detach().cpu(), smoothedPosteriorOverAll[:,0].detach().cpu(), color="yellow")
#            axis[0].scatter(SCALE*grid_indices_here[:,0].detach().cpu(), derivativeSmoothedPosteriorOverAll[:,0].detach().cpu(), color="purple")
#
#            axis[0].scatter(SCALE*environment[:,0].detach().cpu(), smoothedPosteriorInEnvironment[:,0].detach().cpu())
#            axis[1].scatter(SCALE*grid_indices_here[:,worst].detach().cpu(), posterior[:,worst].detach().cpu(), color="gray")
#            axis[1].scatter(SCALE*grid_indices_here[:,worst].detach().cpu(), smoothedPosteriorOverAll[:,worst].detach().cpu(), color="yellow")
#            axis[1].scatter(SCALE*grid_indices_here[:,worst].detach().cpu(), derivativeSmoothedPosteriorOverAll[:,worst].detach().cpu(), color="purple")
#            axis[1].scatter(SCALE*environment[:,worst].detach().cpu(), smoothedPosteriorInEnvironment[:,worst].detach().cpu())
#            Z = max(float(smoothedPosteriorInEnvironment[:,0].max()), float(smoothedPosteriorInEnvironment[:,worst].max()))
#            Y = min(float(smoothedPosteriorInEnvironment[:,0].min()), float(smoothedPosteriorInEnvironment[:,worst].min()))
#            for x, y, z, u in optimizationSequence:
#               print(x[0], x[worst], "P", y[worst], "dP", z[worst], "d2P", u[worst], derivativeSmoothedPosterior[worst], resultDiscrete[worst])
#               axis[0].plot([SCALE*x[0].detach().cpu(), SCALE*x[0].detach().cpu()], [Y,Z], color="yellow")
#               axis[1].plot([SCALE*x[worst].detach().cpu(), SCALE*x[worst].detach().cpu()], [Y,Z], color="yellow")
#            axis[0].plot([SCALE*resultDiscrete1[0].detach().cpu(), SCALE*resultDiscrete1[0].detach().cpu()], [Y,Z], color="orange")
#            axis[1].plot([SCALE*resultDiscrete1[worst].detach().cpu(), SCALE*resultDiscrete1[worst].detach().cpu()], [Y,Z], color="orange")
#            axis[0].plot([SCALE*resultDiscrete[0].detach().cpu(), SCALE*resultDiscrete[0].detach().cpu()], [Y,Z], color="red")
#            axis[1].plot([SCALE*resultDiscrete[worst].detach().cpu(), SCALE*resultDiscrete[worst].detach().cpu()], [Y,Z], color="red")
#            axis[1].plot([SCALE*resultFromIntermediateGrid[worst].detach().cpu(), SCALE*resultFromIntermediateGrid[worst].detach().cpu()], [1.2*Y,1.2*Z], color="purple")
#            savePlot(f"figures/DEBUG_{__file__}.pdf")
#            plt.close()
#            assert False
        if PLOT:
           plt.show()
           plt.close()

        result = resultDiscrete
#        print("FINAL", result*SCALE)
        ctx.save_for_backward(grid, posterior, result)
        return result.detach()*SCALE

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grid, posterior, result = ctx.saved_tensors

        #################
        # g'_j and g''_j are computed as in during the Newton iterations of the forward pass

        # K_{ij} = Kernel_j(\theta_i)
        innerDerivatives = -torch.sin(result.view(GRID,1) - grid.unsqueeze(0)) / (2*(KERNEL_WIDTH**2))
        kernelFunctionAtResult = torch.nn.functional.softmax(torch.cos(result.view(GRID,1) - grid.unsqueeze(0)) / (2*(KERNEL_WIDTH**2)), dim=1)
        kernelFunctionDerivativeI = innerDerivatives * kernelFunctionAtResult
        # This sum is needed repeatedly
        kernelFunctionDerivativeISum = kernelFunctionDerivativeI.sum(dim=1, keepdim=True)
        kernelDerivative = kernelFunctionDerivativeI - kernelFunctionAtResult * kernelFunctionDerivativeISum #kernelDerivative_{ij} = \partial_\theta K_{ij} at \theta_i

        if False:
           plt.close()
           fig, ax = plt.subplots(1, 1)
           for i in range(0,GRID, 10):
               ax.plot(range(GRID), kernelFunctionAtResult[i], linestyle="dotted")
           plt.show()
           plt.close()
   
        if False:
           h = 0.02
           kernelFunctionAtResult_ = torch.nn.functional.softmax(torch.cos(result.view(GRID,1) + h - grid.unsqueeze(0)) / (2*(KERNEL_WIDTH**2)), dim=1)
           for i in range(0, GRID, GRID//180):
               print(kernelFunctionAtResult.size())
               for j in range(0, GRID, GRID//180):
                  print(i, j, kernelFunctionAtResult[i,j], kernelFunctionAtResult_[i,j], "Numerical", (kernelFunctionAtResult_[i,j]-kernelFunctionAtResult[i,j])/h, "Analytical", kernelDerivative[i,j], kernelDerivative[i,j] / ((kernelFunctionAtResult_[i,j]-kernelFunctionAtResult[i,j])/h))
           quit()
   

        derivativeSmoothedPosterior = (kernelDerivative * posterior.t()).sum(dim=1)
        smoothedPosterior = (kernelFunctionAtResult * posterior.t()).sum(dim=1)

        if VERIFY:
           #print(smoothedPosterior)
           h= 0.002
           kernelFunctionAtResult_ = torch.nn.functional.softmax(torch.cos(result.view(GRID,1) + h - grid.unsqueeze(0)) / (2*(KERNEL_WIDTH**2)), dim=1)
           smoothedPosterior_ = (kernelFunctionAtResult_ * posterior.t()).sum(dim=1)
           kernelFunctionAtResult__ = torch.nn.functional.softmax(torch.cos(result.view(GRID,1) - h - grid.unsqueeze(0)) / (2*(KERNEL_WIDTH**2)), dim=1)
           smoothedPosterior__ = (kernelFunctionAtResult__ * posterior.t()).sum(dim=1)
           # in principle, ther derivative at the estimator should be zero
           #print(posterior)
           #print(smoothedPosterior_)
   #        quit()
           for i in range(0, GRID, GRID//180):
   #            print(i, smoothedPosterior[i], smoothedPosterior_[i], "Numerical", (smoothedPosterior_[i]-smoothedPosterior__[i])/(2*h), "Analytical", derivativeSmoothedPosterior[i], derivativeSmoothedPosterior[i] / ((smoothedPosterior_[i]-smoothedPosterior__[i])/(2*h)))
               assert abs(derivativeSmoothedPosterior[i]) < 1e-5
               if abs((smoothedPosterior_[i]-smoothedPosterior__[i])/(2*h)) > 1e-2:
                   print("Warning: Numerical derivative not so close to zero", i, (smoothedPosterior_[i]-smoothedPosterior__[i])/(2*h))
   #        print(grid)
    #       print(result)
   #        quit()




        kernelFunctionSecondDerivativeI = -torch.cos(result.view(GRID,1) - grid.unsqueeze(0)) / (2*(KERNEL_WIDTH**2)) * kernelFunctionAtResult + innerDerivatives.pow(2) * kernelFunctionAtResult
        part1 = kernelFunctionSecondDerivativeI - kernelFunctionDerivativeI * kernelFunctionDerivativeISum
        kernelSecondDerivative = part1 - kernelDerivative * kernelFunctionDerivativeISum - kernelFunctionAtResult * part1.sum(dim=1, keepdim=True)
        # this one looks good
        if False:
           h = 0.02
           innerDerivatives_ = -torch.sin(result.view(GRID,1) + h - grid.unsqueeze(0)) / (2*(KERNEL_WIDTH**2))
           kernelFunctionAtResult_ = torch.nn.functional.softmax(torch.cos(result.view(GRID,1) + h - grid.unsqueeze(0)) / (2*(KERNEL_WIDTH**2)), dim=1)
           kernelFunctionDerivativeI_ = innerDerivatives_ * kernelFunctionAtResult_
           # This sum is needed repeatedly
           kernelFunctionDerivativeISum_ = kernelFunctionDerivativeI_.sum(dim=1, keepdim=True)
           kernelDerivative_ = kernelFunctionDerivativeI_ - kernelFunctionAtResult_ * kernelFunctionDerivativeISum_ #kernelDerivative_{ij} = \partial_\theta K_{ij} at \theta_i
           for i in range(0, GRID, GRID//180):
               for j in range(0, GRID, GRID//180):
                   print(i, j, kernelDerivative[i,j], kernelDerivative_[i,j], "Numerical", (kernelDerivative_[i,j]-kernelDerivative[i,j])/h, "Analytical", kernelSecondDerivative[i,j], kernelSecondDerivative[i,j] / ((kernelDerivative_[i,j]-kernelDerivative[i,j])/h))
   #                assert abs(kernelSecondDerivative[i,j] / ((kernelDerivative_[i,j]-kernelDerivative[i,j])/h) - 1) < .5, abs(kernelSecondDerivative[i,j] / ((kernelDerivative_[i,j]-kernelDerivative[i,j])/h) - 1)
   
   
 #       print(kernelSecondDerivative.size())
#        quit()

        secondDerivativeSmoothedPosterior = (kernelSecondDerivative * posterior.t()).sum(dim=1)
        hessian = secondDerivativeSmoothedPosterior

        if VERIFY:
           h=0.05
           innerDerivatives_ = -torch.sin(result.view(GRID,1) + h - grid.unsqueeze(0)) / (2*(KERNEL_WIDTH**2))
           kernelFunctionAtResult_ = torch.nn.functional.softmax(torch.cos(result.view(GRID,1) + h - grid.unsqueeze(0)) / (2*(KERNEL_WIDTH**2)), dim=1)
           kernelFunctionDerivativeI_ = innerDerivatives_ * kernelFunctionAtResult_
           # This sum is needed repeatedly
           kernelFunctionDerivativeISum_ = kernelFunctionDerivativeI_.sum(dim=1, keepdim=True)
           kernelDerivative_ = kernelFunctionDerivativeI_ - kernelFunctionAtResult_ * kernelFunctionDerivativeISum_ #kernelDerivative_{ij} = \partial_\theta K_{ij} at \theta_i
           derivativeSmoothedPosterior_ = (kernelDerivative_ * posterior.t()).sum(dim=1)
   
           innerDerivatives__ = -torch.sin(result.view(GRID,1) - h - grid.unsqueeze(0)) / (2*(KERNEL_WIDTH**2))
           kernelFunctionAtResult__ = torch.nn.functional.softmax(torch.cos(result.view(GRID,1) - h - grid.unsqueeze(0)) / (2*(KERNEL_WIDTH**2)), dim=1)
           kernelFunctionDerivativeI__ = innerDerivatives__ * kernelFunctionAtResult__
           # This sum is needed repeatedly
           kernelFunctionDerivativeISum__ = kernelFunctionDerivativeI__.sum(dim=1, keepdim=True)
           kernelDerivative__ = kernelFunctionDerivativeI__ - kernelFunctionAtResult__ * kernelFunctionDerivativeISum__ #kernelDerivative__{ij} = \partial_\theta K_{ij} at \theta_i
           derivativeSmoothedPosterior__ = (kernelDerivative__ * posterior.t()).sum(dim=1)
   
           for i in range(0, GRID, GRID//180):
   #            print("d2 P", i, derivativeSmoothedPosterior__[i], derivativeSmoothedPosterior_[i], "Numerical", (derivativeSmoothedPosterior_[i]-derivativeSmoothedPosterior__[i])/(2*h), "Analytical", hessian[i], hessian[i] / ((derivativeSmoothedPosterior_[i]-derivativeSmoothedPosterior__[i])/(2*h)))
               assert (derivativeSmoothedPosterior_[i]-derivativeSmoothedPosterior__[i])/(2*h) <= 0 # should hold around maximum
               if not ( abs(hessian[i] / ((derivativeSmoothedPosterior_[i]-derivativeSmoothedPosterior__[i])/(2*h)) - 1) < 0.2):
                   print("Warning: d2 P might be incorrect", i)
           #quit()

        #################

        # Now using implicit differentiation
        gradient_implicit = - kernelDerivative.t() / hessian
#        print(kernelDerivative.size(), hessian.size(), gradient_implicit.size())
#        quit()
        gradient = grad_output.unsqueeze(0) * gradient_implicit

        return None, gradient*SCALE
