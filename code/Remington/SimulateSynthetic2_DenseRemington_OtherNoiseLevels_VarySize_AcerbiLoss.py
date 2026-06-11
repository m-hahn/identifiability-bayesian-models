import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import torch
from loadRemington import *
from acerbiEstimator import AcerbiEstimator
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
assert P == 2
FOLD_HERE = int(sys.argv[2])
REG_WEIGHT = float(sys.argv[3])
GRID = int(sys.argv[4])
dataSize = int(sys.argv[5])

PRIOR = sys.argv[6] #"UNIFORM"
ENCODING = sys.argv[7] #"STEEPPERIODIC"
NoiseLevels = sys.argv[8] #"25"
SIGMA2_1 = float(sys.argv[9])


SHOW_PLOT = False #(len(sys.argv) < 6) or (sys.argv[5] == "SHOW_PLOT")
DEVICE = 'cpu'
assert GRID == 400





#############################################################
# Part: Exclude responses outside of the stimulus space.
#mask = torch.logical_and((response > 0.0), (response < 3))
#print("Fraction of excluded datapoints", 1-mask.float().mean())
#print("Number of excluded datapoints", (1-mask.float()).sum())
#print("Total datapoints", mask.size())

#target = target[mask]
#response = response[mask]
#Subject = Subject[mask]

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


##############################################################
# Part: Resample the stimulus values
randomState = random.Random(10)


observations_x = MakeFloatTensor([MIN_GRID + randomState.random()*(MAX_GRID-MIN_GRID) for _ in range(dataSize)])
observations_y = MakeFloatTensor([-1 for _ in range(dataSize)])

#############################################################
# Part: Project observed stimuli onto grid
xValues = []
for x in observations_x:
   xValues.append(int( torch.argmin((grid - x).abs())))
xValues = MakeLongTensor(xValues)

stimulus_ = xValues
responses_=observations_y

x_set = sorted(list(set(xValues.cpu().numpy().tolist())))

##############################################
# Part: Initialize the model
parameters = {}
with open("logs/CROSSVALID/RunRemington_Free_Zero.py_0_0_0.1_200.txt", "r") as inFile:
    l = next(inFile)
    l = next(inFile)
    for l in inFile:
      print(l[:20])
      if l.startswith("="):
         break
      z, y = l.strip().split("\t")
      parameters[z.strip()] = MakeFloatTensor(json.loads(y))

parameters["sigma_logit"] = MakeFloatTensor([-1,-2,-3,-4,-5,-6,-7,-8,-9,-10])

shapes = {x : y.size() for x, y in parameters.items()}
init_parameters = parameters







import counterfactualComponents 

counterfactualComponents.setPrior(PRIOR, parameters, grid, MAX_GRID)
counterfactualComponents.setEncoding(ENCODING, parameters, grid, MAX_GRID)

print(grid.size())
print(parameters["prior"].size())
#quit()


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
    assert (x<=1+1e-5).all(), x.max()
    return -(x.pow(2))/2

#############################################################
# Part: Configure the appropriate estimator for minimizing the loss function
assert P >= 2


mixture_of_gaussians_setup = {}
mixture_of_gaussians_setup["pi_l"] = 1
mixture_of_gaussians_setup["mean_1"] = 0.0
mixture_of_gaussians_setup["mean_2"] = 0.0
mixture_of_gaussians_setup["sigma2_1"] = SIGMA2_1
mixture_of_gaussians_setup["sigma2_2"] = 0.1

string_description_of_mixture_of_gaussians_setup = f"pi_{mixture_of_gaussians_setup['pi_l']}_mean1_{mixture_of_gaussians_setup['mean_1']}_mean2_{mixture_of_gaussians_setup['mean_2']}_sigma21_{mixture_of_gaussians_setup['sigma2_1']}_sigma22_{mixture_of_gaussians_setup['sigma2_2']}"

# Part: Import/define the appropriate estimator for minimizing the loss function
AcerbiEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, P=P, MIXTURE_OF_GAUSSIANS_SETUP=mixture_of_gaussians_setup)


#############################################################
# Part: Run the model. This function implements the model itself:
## calculating the likelihood of a given dataset under that model
## and---if the computePredictions argument is set to True--- computes
## the bias and variability of the estimate.
def samplePredictions(stimulus_, sigma_logit, prior, volumeElement, n_samples=100, showLikelihood=False, grid=grid, responses_=None, computePredictions=False, parameters=None, sigma2_stimulus=None, folds=None, lossReduce='mean'):

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
  # Estimator
  assert P == 2
  bayesianEstimate = AcerbiEstimator.apply(grid[grid_indices_here], posterior)
  likelihoods_by_stimulus = likelihoods[:,stimulus_]

  sampled_estimator = bayesianEstimate[torch.distributions.Categorical(likelihoods_by_stimulus.t()).sample()]

  motor_variances = (motor_variance)
  sampled_response = torch.normal(sampled_estimator, STIMULUS_SPACE_VOLUME * motor_variances.sqrt() *torch.ones_like(sampled_estimator))

  # Mixture of estimation and uniform response
  uniform_part = torch.sigmoid(parameters["mixture_logit"])
  choose_uniform = (MakeZeros(sampled_response.size()).uniform_(0,1) < uniform_part)
  sampled_response = torch.where(choose_uniform, MakeZeros(sampled_response.size()).uniform_(MIN_GRID, MAX_GRID).float(), sampled_response.float())

  print(sampled_estimator.min())
  print(sampled_estimator.max())

  return sampled_response.float()

lowestError = 100000

xValues = []
for x in observations_x:
   xValues.append(int( torch.argmin((grid - x).abs())))

yValues = []
for y in observations_y:
   yValues.append(int( torch.argmin((grid - y).abs())))


xValues = MakeLongTensor(xValues)
yValues = MakeLongTensor(yValues)


parameters["sigma2_stimulus"] = MakeFloatTensor([float("inf"),2,3,4,5,6,7,8,9,10])

duration = 0*observations_y

parameters["loss_function_weights_cosine"] = MakeZeros(10)
parameters["loss_function_weights_cosine"].data[0] = -1
parameters["loss_function_weights_sine"] = MakeZeros(10)
parameters["loss_function_weights_sine"].data[0] = 0
parameters["loss_function_weights_sine"].data[1] = 2*.2
parameters["loss_function_weights_sine"].data[2] = 2*(-.13)
coefficients_list_cosine = parameters["loss_function_weights_cosine"].detach().cpu().numpy().tolist()
coefficients_list_sine = parameters["loss_function_weights_sine"].detach().cpu().numpy().tolist()
fourier_speed = list(range(1,len(coefficients_list_cosine)+1))
fourier_speed_tensor = MakeFloatTensor(fourier_speed) / 2

print("SECOND DERIV AT 0", -(fourier_speed_tensor*fourier_speed_tensor*parameters["loss_function_weights_cosine"]).sum())

errors = MakeFloatTensor(list(range(200)))/70 - 100/70
errors = 3*errors
loss_curve_cosine = sum([lambda_Pa * (((torch.cos(Pa * (errors))))) for Pa, lambda_Pa in zip(fourier_speed, coefficients_list_cosine)])
loss_curve_sine = sum([lambda_Pa * (((torch.sin(Pa * (errors))))) for Pa, lambda_Pa in zip(fourier_speed, coefficients_list_sine)])



noiseLevelsByTrial = [int(NoiseLevels[i%len(NoiseLevels)]) for i in range(duration.size()[0])]
random.Random(10).shuffle(noiseLevelsByTrial)
duration = torch.LongTensor(noiseLevelsByTrial)


# encoding is initialized above
prior = torch.nn.functional.softmax(parameters["prior"], dim=0)
volume = torch.softmax(parameters["volume"], dim=0)


observations_y = MakeZeros(xValues.size())
conditions = MakeZeros(xValues.size())
for condition in list([int(x) for x in NoiseLevels]):
   observations_y[duration == condition] = samplePredictions(xValues[duration == condition], parameters["sigma_logit"][condition], prior, volume, n_samples=1000, grid=grid, responses_=observations_y, sigma2_stimulus=0, computePredictions=(True), parameters=parameters)
observations_y = observations_y.float()

SELFID = random.randint(10000, 1000000)

print(xValues)

outPath = f"logs/SIMULATED_REPLICATE/{__file__}_{GRID}_{P}_{NoiseLevels}_N{dataSize}_{PRIOR}_{ENCODING}_{string_description_of_mixture_of_gaussians_setup}.txt"
if os.path.exists(outPath):
   assert False
with open(outPath, "w") as outFile:
    for z, y in parameters.items():
        print(z, "\t", y.detach().cpu().numpy().tolist(), file=outFile)
    print("=======", file=outFile)
    for i in range(xValues.size()[0]):
      print(int(duration[i]), round(float(grid[xValues[i]]),4), round(float(observations_y[i]),4), file=outFile)
