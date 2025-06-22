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
from loadRemington import *
from lpEstimator import LPEstimator
from l1IntervalEstimator import L1Estimator
from mapIntervalEstimator7_DebugFurtherAug import MAPIntervalEstimator
from matplotlib import rc
from util import MakeFloatTensor
from util import MakeLongTensor
from util import MakeZeros
from util import difference
from util import product
from util import savePlot
__file__ = __file__.split("/")[-1]

#rc('font', **{'family':'FreeSans'})

#############################################################
# Part: Collect arguments
OPTIMIZER_VERBOSE = False

P = int(sys.argv[1])
FOLD_HERE = int(sys.argv[2])
REG_WEIGHT = float(sys.argv[3])
GRID = int(sys.argv[4])
assert GRID == 200
dataSize = int(sys.argv[5])
assert dataSize == 9000
PRIOR = sys.argv[6] #"UNIFORM"
ENCODING = sys.argv[7] #"STEEPPERIODIC"
NoiseLevels = sys.argv[8] #"25"

NoiseLevelsList = [int(q) for q in NoiseLevels]

##############################################

assert (observations_x == target).all()
assert (observations_y == response).all()
assert observations_x.size()[0] == dataSize, observations_x.size()

##############################################



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

if P > 1:
  LPEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, P=P,  SCALE=SCALE)
elif P == 1:
  L1Estimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE)
elif P == 0:
   KERNEL_WIDTH = 0.1
   
   averageNumberOfNewtonSteps = 2
   W = 800
   
   MAPIntervalEstimator.set_parameters(KERNEL_WIDTH=KERNEL_WIDTH, MIN_GRID=MIN_GRID, MAX_GRID=MAX_GRID, SCALE=SCALE, W=W, GRID=GRID, OPTIMIZER_VERBOSE=False)
else:
   assert False
#############################################################
# Part: Initialize the model
parameters = {}
with open("logs/CROSSVALID/RunRemington_Lognormal_Zero.py_0_0_0.1_200.txt", "r") as inFile:
    l = next(inFile)
    l = next(inFile)
    for l in inFile:
        print(l[:20])
        if l.startswith("="):
           break
        z, y = l.split("\t")
        parameters[z.strip()] = MakeFloatTensor(json.loads(y))

sigmaLogitFor7 = parameters["sigma_logit"][0]
assert abs(sigmaLogitFor7-(-7)) < 0.5, parameters["sigma_logit"]
parameters["sigma_logit"] = MakeFloatTensor([-1,-2,-3,-4,-5,-6,-7,-8,-9,-10])

shapes = {x : y.size() for x, y in parameters.items()}
init_parameters = parameters

#############################################################
# Part: Specify `similarity` or `difference` functions.

STIMULUS_SPACE_VOLUME = MAX_GRID-MIN_GRID
SENSORY_SPACE_VOLUME = 1
assert SENSORY_SPACE_VOLUME == 1

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
def computeBias(stimulus_, sigma_logit, prior, volumeElement, n_samples=100, showLikelihood=False, grid=grid, responses_=None, parameters=None, computePredictions=False, subject=None, sigma_stimulus=None, sigma2_stimulus=None, duration_=None, folds=None, lossReduce='mean'):

 # Part: Obtain the motor variance by exponentiating the appropriate model parameter
 motor_variance = torch.exp(- parameters["log_motor_var"])
 # Part: Obtain the sensory noise variance. We parameterize it as a fraction of the squared volume of the size of the sensory space
 sigma2 = (SENSORY_SPACE_VOLUME * SENSORY_SPACE_VOLUME)*torch.sigmoid(sigma_logit)
 # Part: Obtain the transfer function as the cumulative sum of the discretized resource allocation (referred to as `volume` element due to the geometric interpretation by Wei&Stocker 2015)
 F = torch.cat([MakeZeros(1), torch.cumsum(volumeElement, dim=0)], dim=0)

 loss = 0
 if True:
  stimulus = stimulus_
  responses = responses_

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
  if P > 1:
    bayesianEstimate = LPEstimator.apply(grid[grid_indices_here], posterior)
  elif P == 1:
    bayesianEstimate = L1Estimator.apply(grid[grid_indices_here], posterior)/GRID * (MAX_GRID-MIN_GRID) + MIN_GRID
    assert bayesianEstimate.max() <= MAX_GRID+0.0001
    assert bayesianEstimate.min() >= MIN_GRID-0.0001
  elif P == 0:
    assert posterior.size() == (GRID, GRID)
#    print((grid[grid_indices_here].size()), posterior.size())
    bayesianEstimate = MAPIntervalEstimator.apply(grid[grid_indices_here], posterior)
  else:
    assert False
  likelihoods_by_stimulus = likelihoods[:,stimulus_]

 # assert bayesianEstimate.max() <= 1.1*MAX_GRID, bayesianEstimate.max()
#  assert bayesianEstimate.min() >= 0.9*MIN_GRID, bayesianEstimate.min()

  ## Compute the motor likelihood
  motor_variances = motor_variance

  uniform_part = torch.sigmoid(parameters["mixture_logit"])
  ## The full likelihood then consists of a mixture of the motor likelihood calculated before, and the uniform
  ## distribution on the full space.
#  motor_likelihoods = (1-uniform_part) * motor_likelihoods + (uniform_part / (GRID) + 0*motor_likelihoods)


  ## If computePredictions==True, compute the bias and variability of the estimate
  if computePredictions:
     bayesianEstimate_byStimulus = bayesianEstimate.unsqueeze(1)
     bayesianEstimate_avg_byStimulus = (bayesianEstimate_byStimulus * likelihoods).sum(dim=0)

     bayesianEstimate_sd_byStimulus = ((bayesianEstimate_byStimulus - bayesianEstimate_avg_byStimulus.unsqueeze(0)).pow(2) * likelihoods).sum(dim=0).sqrt()
     bayesianEstimate_sd_byStimulus = (bayesianEstimate_sd_byStimulus.pow(2) + STIMULUS_SPACE_VOLUME * STIMULUS_SPACE_VOLUME * motor_variances).sqrt()
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

xValues = []
for x in observations_x:
   xValues.append(int( torch.argmin((grid - x).abs())))

yValues = []
for y in observations_y:
   yValues.append(int( torch.argmin((grid - y).abs())))


xValues = MakeLongTensor(xValues)
yValues = MakeLongTensor(yValues)



duration = 0*observations_y
noiseLevelsByTrial = [int(NoiseLevels[i%len(NoiseLevels)]) for i in range(duration.size()[0])]
random.Random(10).shuffle(noiseLevelsByTrial)
duration = torch.LongTensor(noiseLevelsByTrial)


assert ENCODING == "WEBERFIT"

epsilon = torch.exp(parameters["volume"][0])
volume = torch.softmax(-torch.log(epsilon + grid), dim=0).detach()
if PRIOR == "FROMFIT":
  prior = torch.nn.functional.softmax(-((epsilon+grid).log()-parameters["prior_mu"]).pow(2)/(1e-5+torch.exp(-parameters["prior_sigma2"])) - (epsilon+grid).log())
elif PRIOR == "BIFROMFIT": 
   epsilon = math.exp(-1.846029281616211)
   part1 = (-((epsilon+grid).log()-(-0.45)).pow(2)/(1e-5+math.exp(-3.6692357063293457)) - (epsilon+grid).log())
   part2 = (-((epsilon+grid).log()-(0.1)).pow(2)/(1e-5+math.exp(-3.6692357063293457)) - (epsilon+grid).log())
   prior = torch.softmax((torch.softmax(part1, dim=0) + torch.softmax(part2, dim=0)).log(), dim=0)
else:
  assert False



## Pass data to auxiliary script used for retrieving smoothed fits from the dataset

def computeResources(volume, sigma2):
    alphaReciprocal = 1
    return volume / torch.sqrt(sigma2) * GRID / (MAX_GRID-MIN_GRID) * alphaReciprocal

def model(grid):

  lossesBy500 = []
  crossLossesBy500 = []
  noImprovement = 0
  global optim, learning_rate
  for iteration in range(1):
   parameters = init_parameters
   ## In each iteration, recompute
   ## - the resource allocation (called `volume' due to a geometric interpretation)
   ## - the prior

#   volume = SENSORY_SPACE_VOLUME * torch.softmax(parameters["volume"], dim=0)
   ## For interval datasets, we're norming the size of the sensory space to 1

#   prior = torch.nn.functional.softmax(parameters["prior"])

   loss = 0

   ## Create simple visualzation of the current fit once every 500 iterations
   if iteration % 500 == 0:

     assigned = ["PRI", None, "ENC", None, "ATT", "REP", "TOT", None, "VAR"]
     for w, k in enumerate(assigned):
         globals()[k] = w
     notToInclude = [None, "VAR"]
     PAD = [w for w, q in enumerate(assigned) if q in notToInclude]
     gridspec = dict(width_ratios=[1 if x not in notToInclude else .01 for x in assigned])
     figure, axis = plt.subplots(1, len(gridspec["width_ratios"]), figsize=(8,2), gridspec_kw=gridspec)

     plt.tight_layout()
     figure.subplots_adjust(wspace=0.05, hspace=0.0)

     for w in PAD:
       axis[w].set_visible(False)

     grid_cpu = grid.detach().cpu()
#     MASK = torch.logical_and(grid > float(target.min()), grid < float(target.max()))
     axis[PRI].plot(grid_cpu, prior.detach().cpu(), color="gray")
#     axis[PRI].plot([float(sample.min()), float(sample.max())], 2*[0], color="orange")
     x_set = sorted(list(set(xValues.cpu().numpy().tolist())))

   ## Separate train and test/heldout partitions of the data
   trainFolds = [i for i in range(N_FOLDS) if i!=FOLD_HERE]
   testFolds = [FOLD_HERE]

   ## Iterate over the conditions and possibly subjects, if parameters are fitted separately.
   ## In this dataset, there is just one condition, and all parameters are fitted across subjects.
   for DURATION in range(10):
    if DURATION not in NoiseLevelsList:
      continue
    print(DURATION)
    for SUBJECT in [1]:
     ## Run the model at its current parameter values.
     loss_model, bayesianEstimate_model, bayesianEstimate_sd_byStimulus_model, attraction = computeBias(xValues, init_parameters["sigma_logit"][DURATION], prior, volume, n_samples=1000, grid=grid, responses_=observations_y, parameters=parameters, computePredictions=(iteration%100 == 0), subject=None, sigma_stimulus=0, sigma2_stimulus=0, duration_=DURATION, folds=testFolds, lossReduce='sum')
     loss += loss_model
     print("LOSS", loss)

     if iteration % 500 == 0:

       _, bayesianEstimate_model_uniform, _, _ = computeBias(xValues, init_parameters["sigma_logit"][DURATION], 1/GRID + MakeZeros(GRID), volume, n_samples=1000, grid=grid, responses_=observations_y, parameters=parameters, computePredictions=(iteration%100 == 0), subject=None, sigma_stimulus=0, sigma2_stimulus=0, duration_=DURATION, folds=testFolds, lossReduce='sum')

#       Xs = grid[MASK][::1]
#       ys = []
#       sds = []
#       for s in range(0,15):
#         y_set, sd_set = retrieveObservations(x, s)
#         X = MakeFloatTensor(grid[x_set])
#         Y = MakeFloatTensor(y_set)
#         S = MakeFloatTensor(sd_set)
#         M = torch.logical_not(torch.isnan(Y))
#         X = X[M]
#         Y = Y[M]
#         S = S[M]
#         kappa = 0.01
#         KERNEL = torch.nn.functional.softmax(-(X.view(-1,1)-Xs.view(1,-1)).pow(2)/kappa, dim=0)
#         print(X.size(), Xs.size(), Y.size(), KERNEL.size())
#
#         Y_smoothed = (Y.view(-1,1) * KERNEL).sum(dim=0)
#         S_smoothed = (S.view(-1,1) * KERNEL).sum(dim=0)
#
#         ys.append(Y_smoothed)
#         sds.append(S_smoothed)
#       ys = torch.stack(ys, dim=0).mean(dim=0)
#       sds = torch.stack(sds, dim=0).mean(dim=0)
#
       axis[ENC].plot(grid_cpu, computeResources(volume, torch.sigmoid(init_parameters["sigma_logit"][DURATION])).detach().cpu())
       axis[REP].plot(grid_cpu, (bayesianEstimate_model-grid-attraction).detach().cpu())
       axis[ATT].plot(grid_cpu, attraction.detach())
       axis[TOT].plot(grid_cpu, (bayesianEstimate_model-grid).detach().cpu())
       if False:
         axis[HUM].plot(Xs, ys)
       axis[VAR].plot(grid_cpu, bayesianEstimate_sd_byStimulus_model.detach().cpu())
       if False:
         axis[VAH].plot(Xs, sds)

   if iteration % 500 == 0:
#     axis[ENC].set_xlim(0.3, 1.2)
#     axis[PRI].set_xlim(0.3, 1.2)
     if True:
        axis[ENC].set_title("Resources")
        axis[PRI].set_title("Prior")
        axis[REP].set_title("Repulsion")
        axis[ATT].set_title("Attraction")
     if True:
        axis[TOT].set_title("Bias (Model)")
        if False:
          axis[HUM].set_title("Bias (Data)")
        axis[VAR].set_title("Var. (Model)")
        if False:
          axis[VAH].set_title("Var. (Data)")
     for z in range(len(assigned)):
         axis[z].spines['right'].set_visible(False)
         axis[z].spines['top'].set_visible(False)
         axis[z].set_xticks(ticks=[0.5, 1], labels=["0.5", "1.0 s"])
     for i in [REP, ATT, TOT]:
#        axis[i].set_xlim(0.3, 1.2)
        axis[i].set_ylim(-0.4, 0.4)
     for i in [VAR]:
#        axis[i].set_xlim(0.3, 1.2)
        axis[i].set_ylim(0,0.3)
     axis[ENC].set_yticks(ticks=[0, 10])
     axis[PRI].set_yticks(ticks=[0])
     for z in [REP, ATT, TOT]:
         axis[z].set_yticks(ticks=[-0.1, -0.05, 0, 0.05, 0.1], labels=["-0.1 s", "", "0 s", "", "0.1 s"])
     for z in [VAR]:
         axis[z].set_yticks(ticks=[0, 0.02, 0.04, 0.06, 0.08, 0.1], labels=["0 s", "", "", "", "", "0.1 s"])
     for w in range(1,len(assigned)):
         if assigned[w-1] is not None:
             axis[w].set_yticks([])
     savePlot(f"figures/{__file__}_{NoiseLevels}_{PRIOR}_{ENCODING}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.pdf")

     figure, axis = plt.subplots(1, 1, figsize=(2,2))
     figure.tight_layout()
     grid_cpu = grid.detach().cpu()
     #MASK = torch.logical_and(grid > float(target.min()), grid < float(target.max()))
     axis.plot(grid_cpu, prior.detach().cpu())
#     axis.plot([float(sample.min()), float(sample.max())], 2*[0.5*float(prior.max())], color="orange")
     axis.set_xlim(0.3, 1.2)
     savePlot(f"figures/{__file__}_{NoiseLevels}_{PRIOR}_{ENCODING}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}_Prior.pdf")

############################3

model(grid)
