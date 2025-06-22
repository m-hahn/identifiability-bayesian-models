import getObservations
import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import torch
from getObservations import retrieveObservations
from loadRemington import *
from lpEstimator import LPEstimator
from matplotlib import rc
from util import MakeFloatTensor
from util import MakeLongTensor
from util import MakeZeros
from util import difference
from util import product
from util import savePlot
__file__ = __file__.split("/")[-1]

rc('font', **{'family':'FreeSans'})

#############################################################
# Part: Collect arguments
OPTIMIZER_VERBOSE = False

P = int(sys.argv[1])
assert P > 0
FOLD_HERE = int(sys.argv[2])
REG_WEIGHT = float(sys.argv[3])
GRID = int(sys.argv[4])

#############################################################
# Part: Exclude responses outside of the stimulus space.
mask = torch.logical_and((response > 0.0), (response < 3))
print("Fraction of excluded datapoints", 1-mask.float().mean())
print("Number of excluded datapoints", (1-mask.float()).sum())
print("Total datapoints", mask.size())

target = target[mask]
response = response[mask]
Subject = Subject[mask]
observations_x = observations_x[mask]
observations_y = observations_y[mask]

#############################################################
# Part: Partition data into folds. As described in the paper,
# this is done within each subject.
N_FOLDS = 10
assert FOLD_HERE < N_FOLDS
randomGenerator = random.Random(10)

Fold = 0*Subject
for i in range(int(min(Subject)), int(max(Subject))+1):
    trials = [j for j in range(Subject.size()[0]) if Subject[j] == i]
    randomGenerator.shuffle(trials)
    foldSize = int(len(trials)/N_FOLDS)
    for k in range(N_FOLDS):
        Fold[trials[k*foldSize:(k+1)*foldSize]] = k

assert (observations_x == target).all()
assert (observations_y == response).all()

#############################################################
# Part: Set up the discretized grid
MIN_GRID = 0
MAX_GRID = 3
RANGE_GRID = MAX_GRID-MIN_GRID

CIRCULAR = False
INVERSE_DISTANCE_BETWEEN_NEIGHBORING_GRID_POINTS = GRID/(MAX_GRID-MIN_GRID)

grid = MakeFloatTensor([x/GRID * (MAX_GRID-MIN_GRID) for x in range(GRID)]) + MIN_GRID
grid_indices = MakeFloatTensor([x for x in range(GRID)])
def point(p, reference):
    assert not CIRCULAR
    return p
grid_indices_here = MakeLongTensor([[point(y,x) for x in range(GRID)] for y in range(GRID)])

#############################################################
# Part: Project observed stimuli onto grid
xValues = []
for x in observations_x:
   xValues.append(int( torch.argmin((grid - x).abs())))
xValues = MakeLongTensor(xValues)

stimulus_ = xValues
responses_=observations_y

x_set = sorted(list(set(xValues.cpu().numpy().tolist())))

#############################################################
# Part: Configure the appropriate estimator for minimizing the loss function
SCALE=1
LPEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, P=P,  SCALE=SCALE)

#############################################################
# Part: Initialize the model
init_parameters = {}
init_parameters["sigma2_stimulus"] = MakeFloatTensor([0]).view(1)
init_parameters["log_motor_var"] = MakeFloatTensor([3]).view(1)
init_parameters["sigma_logit"] = MakeFloatTensor(1*[-3]).view(1)
init_parameters["mixture_logit"] = MakeFloatTensor([-1]).view(1)
init_parameters["prior"] = MakeZeros(GRID)
init_parameters["volume"] = MakeZeros(GRID)
for _, y in init_parameters.items():
    y.requires_grad = True

FILE = f"logs/CROSSVALID/{__file__}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.txt"
assert "volume" in init_parameters
for _, y in init_parameters.items():
    y.requires_grad = True

#############################################################
# Part: Initialize optimizer.
# The learning rate is a user-specified parameter.
learning_rate = 0.001
optim = torch.optim.SGD([y for _, y in init_parameters.items()], lr=learning_rate, momentum=0.3)

#############################################################
# Part: Specify `similarity` or `difference` functions.

STIMULUS_SPACE_VOLUME = MAX_GRID-MIN_GRID
SENSORY_SPACE_VOLUME = 1

## These are negative squared distances (for interval spaces) or
## trigonometric functions (for circular spaces), with
## some extra factors for numerical purposes.
## Exponentiating a `similarity` function and normalizing
## is equivalent to the Gaussian / von Mises density.
## The purpose of specifying these as `closeness` or `distance`,
## rather than simply calling squared or trigonometric
## functions is to  flexibly reuse the same model code for
## both interval and circular spaces.
def SQUARED_STIMULUS_SIMILARITY(x):
    """ Given a difference x between two stimuli, compute the `similarity` in
    stimulus space. Generally, this is cos(x) for circular spaces and something
    akin to 1-x^2 for interval spaces, possibly defined with additional factors
    to normalize by the size of the space. The resulting values are exponentiated
    and normalized to obtain a Gaussian or von Mises density."""
    assert (x<=STIMULUS_SPACE_VOLUME).all(), x.max()
    return -((x/STIMULUS_SPACE_VOLUME).pow(2))/2
def SQUARED_SENSORY_SIMILARITY(x):
    """ Given a difference x between two stimuli, compute the `similarity` in
    sensory space. Generally, this is cos(x) for circular spaces and something
    akin to 1-x^2 for interval spaces, possibly defined with additional factors
    to normalize by the size of the space. The resulting values are exponentiated
    and normalized to obtain a Gaussian or von Mises density."""
    assert (x<=1).all(), x.max()
    return -(x.pow(2))/2

#############################################################
# Part: Run the model. This function implements the model itself:
## calculating the likelihood of a given dataset under that model
## and---if the computePredictions argument is set to True--- computes
## the bias and variability of the estimate.
def computeBias(stimulus_, sigma_logit, prior, volumeElement, n_samples=100, showLikelihood=False, grid=grid, responses_=None, parameters=None, computePredictions=False, subject=None, sigma_stimulus=None, sigma2_stimulus=None, folds=None, lossReduce='mean'):

 # Part: Obtain the motor variance by exponentiating the appropriate model parameter
 motor_variance = torch.exp(- parameters["log_motor_var"])
 # Part: Obtain the sensory noise variance. We parameterize it as a fraction of the squared volume of the size of the sensory space
 sigma2 = (SENSORY_SPACE_VOLUME * SENSORY_SPACE_VOLUME)*torch.sigmoid(sigma_logit)
 # Part: Obtain the transfer function as the cumulative sum of the discretized resource allocation (referred to as `volume` element due to the geometric interpretation by Wei&Stocker 2015)
 F = torch.cat([MakeZeros(1), torch.cumsum(volumeElement, dim=0)], dim=0)

 if True:
  # Part: Select data for the relevant fold
  folds = MakeLongTensor(folds)
  if subject is not None:
    assert False
  else:
    MASK = (Fold.unsqueeze(0) == folds.unsqueeze(1)).any(dim=0)
    stimulus = stimulus_[MASK]
    responses = responses_[MASK]

  # Part: Apply stimulus noise, if nonzero.
  if sigma2_stimulus > 0:
    ## On this dataset, this is zero, so the
    ## code block will not be used.
    assert False
    stimulus_log_likelihoods = ((SQUARED_STIMULUS_SIMILARITY(grid.unsqueeze(0)-grid.unsqueeze(1)))/(sigma2_stimulus))
    stimulus_likelihoods = torch.nn.Softmax(dim=0)(stimulus_log_likelihoods)

  # Part: Compute sensory likelihoods. Across both interval and
  ## circular stimulus spaces, this amounts to exponentiaring a
  ## `similarity`
  sensory_likelihoods = torch.softmax(((SQUARED_SENSORY_SIMILARITY(F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1)))/(sigma2))  + volumeElement.unsqueeze(1).log(), dim=0)

  # Part: If stimulus noise is nonzero, convolve the likelihood with the
  ## stimulus noise.
  if sigma2_stimulus == 0:
    likelihoods = sensory_likelihoods
  else:
    ## On this dataset, this is zero, so the
    ## code block will not be used.
    assert False
    likelihoods = torch.matmul(sensory_likelihoods, stimulus_likelihoods)

  ## Compute posterior using Bayes' rule. As described in the paper, the posterior is computed
  ## in the discretized stimulus space.
  posterior = prior.unsqueeze(1) * likelihoods.t()
  posterior = posterior / posterior.sum(dim=0, keepdim=True)

  ## Compute the estimator for each m in the discretized sensory space.
  bayesianEstimate = LPEstimator.apply(grid[grid_indices_here], posterior)

  assert bayesianEstimate.max() <= 1.1*MAX_GRID
  assert bayesianEstimate.min() >= 0.9*MIN_GRID

  ## Compute the motor likelihood
  motor_variances = motor_variance

  ## `error' refers to the stimulus similarity between the estimator assigned to each m and
  ## the observations found in the dataset.
  ## The Gaussian or von Mises motor likelihood is obtained by exponentiating and normalizing
  error = (SQUARED_STIMULUS_SIMILARITY((bayesianEstimate.unsqueeze(0) - responses.unsqueeze(1)))/motor_variances.unsqueeze(0))
  ## The log normalizing constants, for each m in the discretized sensory space
  log_normalizing_constants = torch.logsumexp(SQUARED_STIMULUS_SIMILARITY((grid.view(-1,1) - bayesianEstimate.view(1,-1)))/motor_variances.unsqueeze(0), dim=0)
  ## The log motor likelihoods, for each pair of sensory encoding m and observed human response
  log_motor_likelihoods = (error) - log_normalizing_constants.view(1, -1)

  ## Obtaining the motor likelihood by exponentiating.
  motor_likelihoods = torch.exp(log_motor_likelihoods)
  ## Obtain the guessing rate, parameterized via the (inverse) logit transform as described in SI Appendix
  uniform_part = torch.sigmoid(parameters["mixture_logit"])
  ## The full likelihood then consists of a mixture of the motor likelihood calculated before, and the uniform
  ## distribution on the full space.
  motor_likelihoods = (1-uniform_part) * motor_likelihoods + (uniform_part / (GRID) + 0*motor_likelihoods)

  # Now the loss is obtained by marginalizing out m from the motor likelihood
  if lossReduce == 'mean':
    loss = -torch.gather(input=torch.matmul(motor_likelihoods, likelihoods),dim=1,index=stimulus.unsqueeze(1)).squeeze(1).log().mean()
  elif lossReduce == 'sum':
    loss = -torch.gather(input=torch.matmul(motor_likelihoods, likelihoods),dim=1,index=stimulus.unsqueeze(1)).squeeze(1).log().sum()
  else:
    assert False

  ## If computePredictions==True, compute the bias and variability of the estimate
  if computePredictions:
     bayesianEstimate_byStimulus = bayesianEstimate.unsqueeze(1)
     bayesianEstimate_avg_byStimulus = (bayesianEstimate_byStimulus * likelihoods).sum(dim=0)

     bayesianEstimate_sd_byStimulus = ((bayesianEstimate_byStimulus - bayesianEstimate_avg_byStimulus.unsqueeze(0)).pow(2) * likelihoods).sum(dim=0).sqrt()
     bayesianEstimate_sd_byStimulus = (bayesianEstimate_sd_byStimulus.pow(2) + motor_variances).sqrt()
     posteriorMaxima = grid[posterior.argmax(dim=0)]
     posteriorMaxima = (posteriorMaxima.unsqueeze(1) * likelihoods).sum(dim=0)
     encodingBias = (grid.unsqueeze(1) * likelihoods).sum(dim=0)
     attraction = (posteriorMaxima-encodingBias)
  else:
     bayesianEstimate_avg_byStimulus = None
     bayesianEstimate_sd_byStimulus = None
     attraction = None
 if float(loss) != float(loss):
     print("NAN!!!!")
     quit()
 return loss, bayesianEstimate_avg_byStimulus, bayesianEstimate_sd_byStimulus, attraction

## Pass data to auxiliary script used for retrieving smoothed fits from the dataset
getObservations.setData(x_set=x_set, observations_y=observations_y, xValues=xValues, grid=grid, Subject=Subject)

def model(grid):

  lossesBy500 = []
  crossLossesBy500 = []
  noImprovement = 0
  global optim, learning_rate
  averageLossOver100 = [0]
  for iteration in range(10000000):
   parameters = init_parameters
   ## In each iteration, recompute
   ## - the resource allocation (called `volume' due to a geometric interpretation)
   ## - the prior

   volume = SENSORY_SPACE_VOLUME * torch.softmax(parameters["volume"], dim=0)
   ## For interval datasets, we're norming the size of the sensory space to 1

   prior = torch.nn.functional.softmax(parameters["prior"])

   loss = 0

   ## Create simple visualzation of the current fit once every 500 iterations
   if iteration % 500 == 0:
     figure, axis = plt.subplots(2, 3)
     grid_cpu = grid.detach().cpu()
     axis[0,0].scatter(grid_cpu, volume.detach().cpu())
     axis[0,0].plot([grid_cpu[0], grid_cpu[-1]], [0,0])
     axis[1,0].scatter(grid_cpu, prior.detach().cpu())
     axis[1,0].plot([grid_cpu[0], grid_cpu[-1]], [0,0])
     x_set = sorted(list(set(xValues.cpu().numpy().tolist())))

   ## Separate train and test/heldout partitions of the data
   trainFolds = [i for i in range(N_FOLDS) if i!=FOLD_HERE]
   testFolds = [FOLD_HERE]

   ## Iterate over the conditions and possibly subjects, if parameters are fitted separately.
   ## In this dataset, there is just one condition, and all parameters are fitted across subjects.
   for DURATION in range(1):
    for SUBJECT in [1]:
     ## Run the model at its current parameter values.
     loss_model, bayesianEstimate_model, bayesianEstimate_sd_byStimulus_model, attraction = computeBias(xValues, init_parameters["sigma_logit"][DURATION], prior, volume, n_samples=1000, grid=grid, responses_=observations_y, parameters=parameters, computePredictions=(iteration%100 == 0), subject=None, sigma_stimulus=0, sigma2_stimulus=0, folds=trainFolds, lossReduce='sum')
     loss += loss_model

     if iteration % 500 == 0:
       ## Visualization
       y_set, sd_set = retrieveObservations(x, None)
       axis[DURATION,1].plot(grid_cpu, (bayesianEstimate_model-grid).detach().cpu(), color="black")
       axis[DURATION,1].scatter([grid_cpu[q] for q in x_set], y_set, color="green")

       axis[DURATION,2].plot(grid_cpu, bayesianEstimate_sd_byStimulus_model.detach().cpu(), color="black")
       axis[DURATION,2].scatter([grid_cpu[q] for q in x_set], sd_set, color="green")

   if iteration % 500 == 0:
     ## Visualization
     axis[1,0].set_xlim(0.4, 1.2)
     plt.tight_layout()
     savePlot(f"figures/{__file__}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.pdf")
     plt.close()

   if iteration % 500 == 0 and iteration > 0:
     ## Once every 500 iterations, obtain heldout-NLL from the test/heldout partition of the dataset
     crossValidLoss = 0
     loss_model, bayesianEstimate_model, bayesianEstimate_sd_byStimulus_model, attraction = computeBias(xValues, init_parameters["sigma_logit"][DURATION], prior, volume, n_samples=1000, grid=grid, responses_=observations_y, parameters=parameters, computePredictions=(iteration%100 == 0), subject=None, sigma_stimulus=0, sigma2_stimulus=0, folds=testFolds, lossReduce='sum')
     crossValidLoss += loss_model

   ## Part: Regularization
   ## Compute the regularization term.
   ## This is only used for nonparametric components, and is zero for parametric model components.
   regularizer1 = ((init_parameters["prior"][1:] - init_parameters["prior"][:-1]).pow(2).sum())/GRID
   regularizer2 = ((init_parameters["volume"][1:] - init_parameters["volume"][:-1]).pow(2).sum())/GRID
   regularizer_total = regularizer1 + regularizer2

   assert not CIRCULAR

   ## Normalize the loss (i.e., negative log-likelihood) by the number of observations
   loss = loss * (1/observations_x.size()[0])
   if iteration % 10 == 0:
     print("LOSS", loss)
     print(parameters["volume"], parameters["volume"].grad)
   ## Add regularization
   loss = loss + REG_WEIGHT * regularizer_total

   ## Part: A single optimization step
   ## Compute gradients of the loss augmented with regularizatioin
   optim.zero_grad()
   loss.backward()
   maximumGradNorm = []
   largestGradNorm = 0
   ## For monitoring purposes, calculate the size of the gradients
   for w in init_parameters:
     if init_parameters[w].grad is not None:
      maximumGradNorm.append(w)
      gradNormMax = float(init_parameters[w].grad.abs().max())
      maximumGradNorm.append(gradNormMax)
      largestGradNorm = max(largestGradNorm, float(gradNormMax))
      ## Optionally, in order to use SignGD, can now replace each
      ## gradient by an indicator of its sign.
      init_parameters[w].grad.data = torch.sign(init_parameters[w].grad.data)
   if iteration % 10 == 0:
     print(largestGradNorm, maximumGradNorm)

   ## Perform a single gradient descent step
   optim.step()
   ## Here, we found improved convergence using SignGD and annealing the step size in intervals of 100 steps
   averageLossOver100[-1] += float(loss) / 100
   if iteration % 10 == 0:
     print("Run for ", iteration, "Iterations.", averageLossOver100[-3:-1], loss.item(), init_parameters["sigma_logit"].detach().cpu().numpy().tolist(), "mixture_logit", init_parameters["mixture_logit"].detach().cpu().numpy().tolist(), "log_motor_var", init_parameters["log_motor_var"].detach().cpu().numpy().tolist(), learning_rate, sys.argv)
   if iteration % 100 == 0 and iteration > 0:
       averageLossOver100.append(0)
   ## Part: Monitor convergence of losses, save fitted results, and adjust step size
   if iteration % 500 == 0 and iteration > 0:
       ## Record losses
       lossesBy500.append(float(loss))
       crossLossesBy500.append(float(crossValidLoss))
       ## If loss has decreased, save the current fit
       if len(lossesBy500) == 1 or float(loss) <= min(lossesBy500):
        with open(f"losses/{FILE.replace('logs/CROSSVALID/', '')}.txt", "w") as outFile:
           print(float(crossValidLoss), file=outFile)
        with open(f"logs/CROSSVALID/{__file__}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.txt", "w") as outFile:
           print(float(loss), "CrossValid", float(crossValidLoss), "CrossValidLossesBy500", " ".join([str(q) for q in crossLossesBy500]), file=outFile)
           print(iteration, "LossesBy500", " ".join([str(q) for q in lossesBy500]), file=outFile)
           for z, y in init_parameters.items():
               print(z, "\t", y.detach().cpu().numpy().tolist(), file=outFile)
           print("========", file=outFile)
           print("\t".join([str(q) for q in maximumGradNorm]), file=outFile)
       ## If the gradients are close to zero, fitting can stop
       if largestGradNorm < 1e-5:
          print("Converged to stationary point")
          break
       ## If the loss has not improved in a while, decay the step size
   if iteration % 100 == 0 and iteration > 0:
       if len(averageLossOver100) > 2 and float(averageLossOver100[-2]) >= averageLossOver100[-3]-1e-5:
         learning_rate *= 0.3
         optim = torch.optim.SGD([y for _, y in init_parameters.items()], lr=learning_rate, momentum=0.3)
   if iteration % 100 == 0 and iteration > 0:
       if len(averageLossOver100) > 2 and float(averageLossOver100[-2]) >= min(averageLossOver100[:-2]):
         noImprovement += 1
       else:
         noImprovement = 0
       if noImprovement >= 10:
           print(float(averageLossOver100[-2]), min(averageLossOver100[:-2]), averageLossOver100)
           print("Stopping")
           break

############################3

model(grid)
