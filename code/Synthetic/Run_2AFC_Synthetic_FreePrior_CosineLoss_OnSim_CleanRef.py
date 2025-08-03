import getObservations
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import torch
from cosineEstimator6 import CosineEstimator
from getObservations import retrieveObservations
from loadGardelle import *
from matplotlib import rc
from scipy.io import loadmat
from util import MakeFloatTensor
from util import MakeLongTensor
from util import MakeZeros
from util import computeCenteredMean
from util import computeCircularMean
from util import computeCircularMeanWeighted
from util import computeCircularSD
from util import computeCircularSDWeighted
from util import makeGridIndicesCircular
from util import product
from util import savePlot
from util import toFactor

__file__ = __file__.split("/")[-1]
rc('font', **{'family':'FreeSans'})

OPTIMIZER_VERBOSE = False




P = int(sys.argv[1])
assert P > 0
FOLD_HERE = int(sys.argv[2])
REG_WEIGHT = float(sys.argv[3])
GRID = int(sys.argv[4])
SHOW_PLOT = False #(len(sys.argv) < 6) or (sys.argv[5] == "SHOW_PLOT")
DEVICE = 'cuda'
FIT = sys.argv[5] #f"SimulateSynthetic_Parameterized.py_8_12345_UNIFORM_UNIFORM.txt"

#assert "UNIMODAL" not in FIT or "SimulateSynthetic_Parameterized.py_8_12345_UNIMODAL2_UNIFORM.txt" in FIT
#assert "SHIFTED" in FIT

#if len( glob.glob(f"losses/{__file__.replace('_VIZ', '')}_{FIT}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.txt.txt")) > 0:
#   assert False, f"losses/{__file__.replace('_VIZ', '')}_{FIT}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.txt.txt"

#noiseConditions = "12345"
#assert "_"+noiseConditions+"_" in FIT


# Helper Functions dependent on the device

##############################################

with open(f"logs/SIMULATED_REPLICATE/{FIT}", "r") as inFile:
  data = [z.split(" ") for z in inFile.read().strip().split("=======\n")[1].split("\n")         ]

duration__, duration_reference__, sample__, sample_reference__, responses__, responses_probs__ = zip(*data)
duration__ = MakeLongTensor([int(q) for q in duration__])
assert len(set(duration_reference__)) == 1, duration_reference__
DURATION_reference = int(duration_reference__[0])
duration = MakeLongTensor(duration__)
sample_reference = MakeFloatTensor([float(q) for q in sample_reference__])
sample = MakeFloatTensor([float(q) for q in sample__])
responses = MakeFloatTensor([float(q) for q in responses__])
# Store observations
observations_x = sample
observations_x_reference = sample_reference
observations_y = responses

assert observations_x.size() == observations_y.size(), observations_x.size()
assert observations_x.size() == duration.size(), duration.size()
# Assign folds
#############################################################
# Part: Partition data into folds. As described in the paper,
# this is done within each subject.
N_FOLDS = 10
assert FOLD_HERE < N_FOLDS
randomGenerator = random.Random(10)

Fold = [i%10 for i in range(observations_x.size()[0])]
randomGenerator.shuffle(Fold)
Fold = MakeLongTensor(Fold)

##############################################
# Set up the discretized grid
MIN_GRID = 0
MAX_GRID = 360

CIRCULAR = True
INVERSE_DISTANCE_BETWEEN_NEIGHBORING_GRID_POINTS = GRID/(MAX_GRID-MIN_GRID)

grid = MakeFloatTensor([x/GRID * (MAX_GRID-MIN_GRID) for x in range(GRID)]) + MIN_GRID
grid_indices = MakeFloatTensor([x for x in range(GRID)])
grid, grid_indices_here = makeGridIndicesCircular(GRID, MIN_GRID, MAX_GRID)
assert grid_indices_here.max() >= GRID, grid_indices_here.max()

# Project observed stimuli onto grid
xValues = []
for x in observations_x:
   xValues.append(int( torch.argmin((grid - x).abs())))
xValues = MakeLongTensor(xValues)

xValues_reference = []
for x in observations_x_reference:
   xValues_reference.append(int( torch.argmin((grid - x).abs())))
xValues_reference = MakeLongTensor(xValues_reference)


stimulus_ = xValues
responses_=observations_y

x_set = sorted(list(set(xValues.cpu().numpy().tolist())))

##############################################
# Initialize the model
init_parameters = {}
init_parameters["sigma2_stimulus"] = MakeFloatTensor([0]).view(1)
init_parameters["log_motor_var"] = MakeFloatTensor([0]).view(1)
init_parameters["sigma_logit"] = MakeFloatTensor(10*[-3]).view(10)
init_parameters["mixture_logit"] = MakeFloatTensor([-1]).view(1)
init_parameters["prior"] = MakeZeros(GRID)
init_parameters["volume"] = MakeZeros(GRID)
for _, y in init_parameters.items():
    y.requires_grad = True

# Initialize optimizer.
# The learning rate is a user-specified parameter.
learning_rate=.1
optim = torch.optim.SGD([y for _, y in init_parameters.items()], lr=learning_rate)

##############################################
# Part: Specify `similarity` or `difference` functions.

STIMULUS_SPACE_VOLUME = MAX_GRID-MIN_GRID
SENSORY_SPACE_VOLUME = 2*math.pi

# Part: Specify `similariy` or `difference` functions.
## These are negative squared distances (for interval spaces) or
## trigonometric functions (for circular spaces), with
## some extra factors for numerical purposes.
## Exponentiating a `similarity` function and normalizing
## is equivalent to the Gaussian / von Mises density.
## The purpose of specifying these as `closeness` or `distance`,
## rather than simply calling squared or trigonometric
## functions is to  flexibly reuse the same model code for
## both interval and circular spaces.
def SQUARED_STIMULUS_DIFFERENCE(x):
    return torch.sin(math.pi*x/180)
def SQUARED_STIMULUS_SIMILARITY(x):
    """ Given a difference x between two stimuli, compute the `similarity` in
    stimulus space. Generally, this is cos(x) for circular spaces and something
    akin to 1-x^2 for interval spaces, possibly defined with additional factors
    to normalize by the size of the space. The resulting values are exponentiated
    and normalized to obtain a Gaussian or von Mises density."""
    return torch.cos(math.pi*x/180)
def SQUARED_SENSORY_SIMILARITY(x):
    """ Given a difference x between two stimuli, compute the `similarity` in
    sensory space. Generally, this is cos(x) for circular spaces and something
    akin to 1-x^2 for interval spaces, possibly defined with additional factors
    to normalize by the size of the space. The resulting values are exponentiated
    and normalized to obtain a Gaussian or von Mises density."""
    return torch.cos(x)
def SQUARED_SENSORY_DIFFERENCE(x):
    return torch.sin(x)

#############################################################
# Part: Configure the appropriate estimator for minimizing the loss function
assert P >= 2

# Part: Import/define the appropriate estimator for minimizing the loss function
CosineEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, P=P, SQUARED_SENSORY_DIFFERENCE=SQUARED_SENSORY_DIFFERENCE, SQUARED_SENSORY_SIMILARITY=SQUARED_SENSORY_SIMILARITY)

#############################################################
# Part: Run the model. This function implements the model itself:
## calculating the likelihood of a given dataset under that model
## and---if the computePredictions argument is set to True--- computes
## the bias and variability of the estimate.
def computeBias(stimulus_, stimulus_reference_, sigma_logit, sigma_logit_reference, prior, volumeElement, n_samples=100, showLikelihood=False, grid=grid, responses_=None, parameters=None, computePredictions=False, subject=None, sigma_stimulus=None, sigma2_stimulus=None, duration_=None, folds=None, lossReduce='mean'):

 # Part: Obtain the motor variance by exponentiating the appropriate model parameter
 motor_variance = torch.exp(- parameters["log_motor_var"])
 # Part: Obtain the sensory noise variance.
 sigma2 = 4*torch.sigmoid(sigma_logit)
 sigma2_reference = 4*torch.sigmoid(sigma_logit_reference)
 # Part: Obtain the transfer function as the cumulative sum of the discretized resource allocation (referred to as `volume` element due to the geometric interpretation by Wei&Stocker 2015)
 F = torch.cat([MakeZeros(1), torch.cumsum(volumeElement, dim=0)], dim=0)

 if True:
  # Part: Select data for the relevant fold
  folds = MakeLongTensor(folds)
  if subject is not None:
    assert False
  else:
    assert duration.view(-1).size() == Fold.view(-1).size(), (duration.size(), Fold.size())
    MASK = torch.logical_and(duration==duration_, (Fold.unsqueeze(0) == folds.unsqueeze(1)).any(dim=0))
    stimulus = stimulus_[MASK]
    stimulus_reference = stimulus_reference_[MASK]
    responses = responses_[MASK]
  if stimulus.view(-1).size()[0] == 0:
    print("Warning", "no data for this condition in this fold", folds, duration_)
    return 0

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
  sensory_likelihoods_reference = MakeFloatTensor(torch.eye(GRID).numpy().tolist()) # torch.softmax(1000000000*((SQUARED_SENSORY_SIMILARITY(F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1))))  + volumeElement.unsqueeze(1).log(), dim=0).detach()
  #print(sensory_likelihoods_reference)
  #quit()

  # Part: If stimulus noise is nonzero, convolve the likelihood with the
  ## stimulus noise.
  if sigma2_stimulus == 0:
    likelihoods = sensory_likelihoods
    likelihoods_reference = sensory_likelihoods_reference
  else:
    ## On this dataset, this is zero, so the
    ## code block will not be used.
    assert False
    likelihoods = torch.matmul(sensory_likelihoods, stimulus_likelihoods)

  ## Compute posterior using Bayes' rule. As described in the paper, the posterior is computed
  ## in the discretized stimulus space.
  posterior = prior.unsqueeze(1) * likelihoods.t()
  posterior = posterior / posterior.sum(dim=0, keepdim=True)

  posterior_reference = likelihoods_reference.t()


  ## Compute the estimator for each m in the discretized sensory space.
  bayesianEstimate = MAX_GRID/GRID * CosineEstimator.apply(grid_indices_here, posterior)
 # print(grid)
#  print(bayesianEstimate)
  bayesianEstimate_reference = grid #MAX_GRID/GRID * CosineEstimator.apply(grid_indices_here, posterior_reference)
  #quit()

  INVERSE_BANDWIDTH_FACTOR = 500

  kernel = torch.softmax(((INVERSE_BANDWIDTH_FACTOR)*torch.cos(math.pi/180 * (bayesianEstimate.view(-1,1)-grid.view(1,-1)))), dim=1) # for each observation, we have a little hill
  density = torch.matmul(kernel.t(), likelihoods) # each hill is weighted by the probability, all are added up
  density = density / density.sum(dim=0, keepdim=True) # we then normalize the density as if it were a discrete distribution on the grid
#  print(density[:,3])
 # quit()

  kernel_reference = torch.softmax(((INVERSE_BANDWIDTH_FACTOR)*torch.cos(math.pi/180 * (bayesianEstimate_reference.view(-1,1)-grid.view(1,-1)))), dim=1)
  density_reference = torch.matmul(kernel_reference.t(), likelihoods_reference)
  density_reference = density_reference / density_reference.sum(dim=0, keepdim=True)



  comparison1 = torch.sin((grid.unsqueeze(0)-grid.unsqueeze(1))/180*math.pi) > 0 #((grid.unsqueeze(0)-grid.unsqueeze(1)) % GRID > (GRID/2)).float()
  comparison2 = torch.sin((grid.unsqueeze(0)-grid.unsqueeze(1))/180*math.pi) == 0
  comparison = comparison1 + .5 * comparison2


  probabilityOfGivingOne = torch.einsum("ij,is,jt->st", comparison, density_reference, density)
#  print(density.size())
#  comparison = ((bayesianEstimate.unsqueeze(0)-bayesianEstimate_reference.unsqueeze(1)) % GRID > (GRID/2)).float()
#  print(likelihoods, likelihoods[:,5])
 # quit()
  #print(((bayesianEstimate.unsqueeze(0)-bayesianEstimate_reference.unsqueeze(1)))[:,5])
  #print(comparison[:,5])
#  quit()
#  probabilityOfGivingOne = torch.einsum("ij,is,jt->st", comparison, likelihoods_reference, likelihoods)
  #print(probabilityOfGivingOne[:,5])
  #quit()
  #print(probabilityOfGivingOne)
  probabilityOfGivingOne = probabilityOfGivingOne[stimulus_reference, stimulus]
  #print(probabilityOfGivingOne)
  #print(comparison.size(), probabilityOfGivingOne.size(), responses_.size())
  #quit()
  loss = torch.where(responses>0, probabilityOfGivingOne.log(), (1-probabilityOfGivingOne).log())

  # Now the loss is obtained by marginalizing out m from the motor likelihood
  if lossReduce == 'mean':
    loss = -loss.mean()
  elif lossReduce == 'sum':
    loss = -loss.sum()
  else:
    assert False


#  sampled_response = torch.where(sampled_estimator<sampled_estimator_reference, 1, -1)


 if float(loss) != float(loss):
     print("NAN!!!!")
     quit()
 return loss

## Pass data to auxiliary script used for retrieving smoothed fits from the dataset
getObservations.setData(x_set=x_set, observations_y=observations_y, xValues=xValues, duration=duration, grid=grid)

def model(grid):
  lossesBy500 = []
  crossLossesBy500 = []
  noImprovement = 0
  global optim, learning_rate
  for iteration in range(10000000):
   parameters = init_parameters

   ## In each iteration, recompute
   ## - the resource allocation (called `volume' due to a geometric interpretation)
   ## - the prior

   # Define prior and resource allocation
   volume = SENSORY_SPACE_VOLUME * torch.softmax(parameters["volume"], dim=0)
   prior = torch.softmax(parameters["prior"], dim=0)
#   print(parameters["volume"].grad)
 #  print(parameters["prior"].grad)

   loss = 0

   if iteration % 500 == 0:

     x_set = sorted(list(set(xValues.cpu().numpy().tolist())))

   trainFolds = [i for i in range(N_FOLDS) if i!=FOLD_HERE]
   testFolds = [FOLD_HERE]

   for DURATION in range(1,10):
    if (duration == DURATION).long().sum() == 0:
       continue
    for SUBJECT in [1]:
     loss_model  = computeBias(xValues, xValues_reference,  init_parameters["sigma_logit"][DURATION], init_parameters["sigma_logit"][DURATION_reference], prior, volume, n_samples=1000, grid=grid, responses_=observations_y, parameters=parameters, computePredictions=(iteration%100 == 0), subject=None, sigma_stimulus=0, sigma2_stimulus=0, duration_=DURATION, folds=trainFolds, lossReduce='sum')
     loss += loss_model

     if iteration % 500 == 0:
       y_set, sd_set = retrieveObservations(x, None, DURATION)

   if iteration % 500 == 0:

     crossValidLoss = 0
     for DURATION in range(1,10):
      if (duration == DURATION).long().sum() == 0:
         continue
      for SUBJECT in [1]:

       loss_model = computeBias(xValues, xValues_reference,  init_parameters["sigma_logit"][DURATION], init_parameters["sigma_logit"][DURATION_reference], prior, volume, n_samples=1000, grid=grid, responses_=observations_y, parameters=parameters, computePredictions=(iteration%500 == 0), subject=None, sigma_stimulus=0, sigma2_stimulus=0, duration_=DURATION, folds=testFolds, lossReduce='sum')
       crossValidLoss += loss_model

   # Regularization
   ## Part: Regularization
   ## Compute the regularization term.
   ## This is only used for nonparametric components, and is zero for parametric model components.
   regularizer1 = ((init_parameters["volume"][1:] - init_parameters["volume"][:-1]).pow(2).sum() + (init_parameters["volume"][0] - init_parameters["volume"][-1]).pow(2))/GRID
   regularizer2 = ((init_parameters["prior"][1:] - init_parameters["prior"][:-1]).pow(2).sum() + (init_parameters["prior"][0] - init_parameters["prior"][-1]).pow(2))/GRID
   regularizer_total = regularizer1 + regularizer2

   loss = loss * (5/observations_y.size()[0])
   loss = loss + REG_WEIGHT * regularizer_total
   ## Part: A single optimization step
   optim.zero_grad()
   loss.backward()
   optim.step()
   if iteration % 10 == 0:
     print(iteration, loss, init_parameters["sigma_logit"], init_parameters["mixture_logit"], init_parameters["log_motor_var"], torch.exp(-init_parameters["sigma2_stimulus"]))

   if iteration % 500 == 0 and iteration > 0:
       lossesBy500.append(float(loss))
       crossLossesBy500.append(float(crossValidLoss))
       if len(lossesBy500) > 0 and float(crossValidLoss) <= min(crossLossesBy500):
        with open(f"losses/{__file__.replace('_VIZ', '')}_{FIT}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.txt.txt", "w") as outFile:
            print(float(crossValidLoss), file=outFile)
        with open(f"logs/CROSSVALID/{__file__}_{FIT}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.txt", "w") as outFile:
           print(float(loss), "CrossValid", float(crossValidLoss), "CrossValidLossesBy500", " ".join([str(q) for q in crossLossesBy500]), file=outFile)
           print(iteration, "LossesBy500", " ".join([str(q) for q in lossesBy500]), file=outFile)
           for z, y in init_parameters.items():
               print(z, "\t", y.detach().cpu().numpy().tolist(), file=outFile)
       if len(lossesBy500) > 1 and float(loss) > lossesBy500[-2] - 1e-3:
         learning_rate *= 0.8
         optim = torch.optim.SGD([y for _, y in init_parameters.items()], lr=learning_rate)
       if len(lossesBy500) > 1 and float(loss) > min(lossesBy500[:-1]) - 1e-5:
         noImprovement += 1
       else:
         noImprovement = 0
       if noImprovement >= 5:
           print("Stopping")
           break

############################3

model(grid)
