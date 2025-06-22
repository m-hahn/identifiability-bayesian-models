import getObservations
import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy
import scipy.stats
import sys
import torch
from cosineEstimator6 import CosineEstimator
from getObservations import retrieveObservations
from loadGardelle import *
from matplotlib import rc
from mapCircularEstimator10 import MAPCircularEstimator
from matplotlib.backends.backend_pdf import PdfPages
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
from util import sech
from util import sign
from util import toFactor

__file__ = __file__.split("/")[-1]
rc('font', **{'family':'FreeSans'})

#############################################################
# Part: Collect arguments
OPTIMIZER_VERBOSE = False

P = int(sys.argv[1])
assert P % 2 == 0
FOLD_HERE = int(sys.argv[2])
REG_WEIGHT = float(sys.argv[3])
GRID = int(sys.argv[4])
dataSize = int(sys.argv[5])

PRIOR = sys.argv[6] #"UNIFORM"
ENCODING = sys.argv[7] #"STEEPPERIODIC"
NoiseLevels = sys.argv[8] #"25"


SHOW_PLOT = False #(len(sys.argv) < 6) or (sys.argv[5] == "SHOW_PLOT")
DEVICE = 'cpu'

FILE = f"logs/CROSSVALID/{__file__.replace('_VIZ', '')}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.txt"

# Helper Functions dependent on the device

##############################################

# Store observations
assert (observations_x == sample).all()
assert (observations_y == responses).all()




multitude = 1 + dataSize // observations_x.size()[0]
observations_x = torch.cat(multitude * [observations_x], dim=0)[:dataSize]
observations_y = torch.cat(multitude * [observations_y], dim=0)[:dataSize]
sample = torch.cat(multitude * [sample], dim=0)[:dataSize]
responses = torch.cat(multitude * [responses], dim=0)[:dataSize]
Subject = torch.cat(multitude * [Subject], dim=0)[:dataSize]
duration = torch.cat(multitude * [duration], dim=0)[:dataSize]


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

stimulus_ = xValues
responses_=observations_y




x_set = sorted(list(set(xValues.cpu().numpy().tolist())))

##############################################
# Initialize the model
parameters = {}
with open(f"logs/CROSSVALID/RunGardelle_FreePrior_CosineLoss.py_8_0_{REG_WEIGHT}_{GRID}.txt", "r") as inFile:
    l = next(inFile)
    l = next(inFile)
    for l in inFile:
      print(l[:20])
      z, y = l.strip().split("\t")
      parameters[z.strip()] = MakeFloatTensor(json.loads(y))

shapes = {x : y.size() for x, y in parameters.items()}

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
#assert P >= 2
if P >= 2:
  CosineEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, P=P, SQUARED_SENSORY_DIFFERENCE=SQUARED_SENSORY_DIFFERENCE, SQUARED_SENSORY_SIMILARITY=SQUARED_SENSORY_SIMILARITY)


# these parameters are chosen to avoid numerical problems / NaNs. No evidence that they hurt NLL.
SCALE = 10
KERNEL_WIDTH = 20

MAPCircularEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, KERNEL_WIDTH=KERNEL_WIDTH, SCALE=SCALE, MIN_GRID=MIN_GRID, MAX_GRID=MAX_GRID)


#############################################################
# Part: Run the model. This function implements the model itself:
## calculating the likelihood of a given dataset under that model
## and---if the computePredictions argument is set to True--- computes
## the bias and variability of the estimate.
def samplePredictions(stimulus_, sigma_logit, prior, volumeElement, n_samples=100, showLikelihood=False, grid=grid, responses_=None, parameters=None, computePredictions=False, subject=None, sigma_stimulus=None, sigma2_stimulus=None, duration_=None, folds=None):
 motor_variance = torch.exp(- parameters["log_motor_var"])
 # Part: Obtain the sensory noise variance.
 sigma2 = 4*torch.sigmoid(sigma_logit)
 # Part: Obtain the transfer function as the cumulative sum of the discretized resource allocation (referred to as `volume` element due to the geometric interpretation by Wei&Stocker 2015)
 F = torch.cat([MakeZeros(1), torch.cumsum(volumeElement, dim=0)], dim=0)

 loss = 0
 if True:

  if sigma2_stimulus > 0:
    assert False
    stimulus_log_likelihoods = ((SQUARED_STIMULUS_SIMILARITY(grid.unsqueeze(0)-grid.unsqueeze(1)))/(sigma2_stimulus))
    stimulus_likelihoods = torch.nn.Softmax(dim=0)(stimulus_log_likelihoods)

  # Part: Compute sensory likelihoods. Across both interval and
  ## circular stimulus spaces, this amounts to exponentiaring a
  ## `similarity`
  sensory_likelihoods = torch.softmax(((SQUARED_SENSORY_SIMILARITY(F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1)))/(sigma2))  + volumeElement.unsqueeze(1).log(), dim=0)

  if sigma2_stimulus == 0:
    likelihoods = sensory_likelihoods
  else:
    assert False
    likelihoods = torch.matmul(sensory_likelihoods, stimulus_likelihoods)

  # Compute posterior
  posterior = prior.unsqueeze(1) * likelihoods.t()
  posterior = posterior / posterior.sum(dim=0, keepdim=True)

  # Estimator
  if P > 0:
    bayesianEstimate = CosineEstimator.apply(grid_indices_here, posterior)
  else:
    bayesianEstimate = MAPCircularEstimator.apply(grid_indices_here, posterior)
  likelihoods_by_stimulus = likelihoods[:,stimulus_]
  print(likelihoods_by_stimulus.size())
  sampled_estimator = bayesianEstimate[torch.distributions.Categorical(likelihoods_by_stimulus.t()).sample()] * 360 / GRID
  print(sampled_estimator)
  print(sampled_estimator.size())

  motor_noise = scipy.stats.vonmises.rvs(1/float(motor_variance), size=stimulus_.size()[0]) * 180 / math.pi
  print(motor_noise.min(), motor_noise.max())
  sampled_response = (sampled_estimator + motor_noise) % 360

  # Mixture of estimation and uniform response
  uniform_part = torch.sigmoid(parameters["mixture_logit"])
  choose_uniform = (MakeZeros(sampled_response.size()).uniform_(0,1) < uniform_part)
  sampled_response = torch.where(choose_uniform, MakeZeros(sampled_response.size()).uniform_(0,360).float(), sampled_response.float())

  print(sampled_estimator.min())
  print(sampled_estimator.max())

  return sampled_response.float()

lowestError = 1000

xValues = []
for x in observations_x:
   xValues.append(int( torch.argmin((grid - x).abs())))

yValues = []
for y in observations_y:
   yValues.append(int( torch.argmin((grid - y).abs())))

xValues = MakeLongTensor(xValues)
yValues = MakeLongTensor(yValues)

additionalSigmaLogits = [-4.8, -5.3, -5.8, -6.3]

parameters["sigma_logit"] = MakeFloatTensor(parameters["sigma_logit"].numpy().tolist() + additionalSigmaLogits)

#[-3.0, 0.8670228123664856, -1.4027767181396484, -3.24409556388855, -3.7997844219207764, -4.360845565795898]

if PRIOR == "UNIFORM":
   parameters["prior"] = 0*grid
elif PRIOR == "PERIODIC":
   parameters["prior"] = (2-torch.sin(grid/MAX_GRID*2*math.pi).abs()).log()
elif PRIOR == "SHIFTED":
   parameters["prior"] = (2-torch.sin(grid/MAX_GRID*2*math.pi+math.pi/2).abs()).log()
elif PRIOR == "STEEPSHIFTED":
   parameters["prior"] = (1.5-torch.sin(grid/MAX_GRID*2*math.pi+math.pi/2).abs()).log()
elif PRIOR == "STEEPPERIODIC":
   parameters["prior"] = (1.5-torch.sin(grid/MAX_GRID*2*math.pi).abs()).log()
elif PRIOR == "SQUAREPERIODIC":
   parameters["prior"] = 2 * (2-torch.sin(grid/MAX_GRID*2*math.pi).abs()).log()
elif PRIOR == "SQUARESTEEPPERIODIC":
   parameters["prior"] = 2 * (1.5-torch.sin(grid/MAX_GRID*2*math.pi).abs()).log()
elif PRIOR == "SQRTSTEEPPERIODIC":
   parameters["prior"] = .5 * (1.5-torch.sin(grid/MAX_GRID*2*math.pi).abs()).log()
elif PRIOR == "BIMODAL":
   parameters["prior"] = .2 * torch.cos(grid/MAX_GRID*2*math.pi+0.25*math.pi) + .4 * torch.cos(2*grid/MAX_GRID*2*math.pi+0.75*math.pi)
#   figure, axis = plt.subplots(1, 1)
#   axis.plot(grid.detach(), torch.softmax(parameters["prior"].detach(), dim=0))
#   axis.plot(grid.detach(), 0*grid.detach())
#   plt.show()
#   plt.close()
elif PRIOR == "SMOOTHUNIMODAL":
   parameters["prior"] = torch.cos(grid/MAX_GRID*2*math.pi+math.pi) / 3
elif PRIOR == "UNIMODAL2":
   parameters["prior"] = .2 * torch.cos(grid/MAX_GRID*2*math.pi+math.pi)
elif PRIOR == "UNIMODAL":
   parameters["prior"] = .5 * torch.cos(grid/MAX_GRID*2*math.pi+math.pi)
#   figure, axis = plt.subplots(1, 1)
#   axis.scatter(grid.detach(), torch.softmax(parameters["prior"].detach(), dim=0))
#   axis.scatter(grid.detach(), 0*grid.detach())
#   plt.show()
#   plt.close()
elif PRIOR == "FITTED":
   pass
elif PRIOR.startswith("FOURIER"):
  _, seed = PRIOR.split("_")
  import random
  rstate = random.Random(int(seed))
  frequencies = torch.LongTensor(range(5))
  sines = torch.sin(grid.view(1,-1)*frequencies.view(-1,1) * 2*math.pi/MAX_GRID)
  cosines = torch.cos(grid.view(1,-1)*frequencies.view(-1,1) * 2*math.pi/MAX_GRID)
  basis = torch.cat([sines, cosines], dim=0)
  coefficients = torch.FloatTensor([rstate.random()-0.5 for _ in range(10)])  #/ torch.cat([frequencies, frequencies], dim=0).clamp(min=1)
  parameters["prior"] = (basis*coefficients.unsqueeze(1)).sum(dim=0)
#  figure, axis = plt.subplots(1, 1)
#  axis.scatter(grid.detach(), torch.softmax(parameters["prior"].detach(), dim=0))
#  axis.scatter(grid.detach(), 0*grid.detach())
#  plt.show()
#  plt.close()
else:
   assert False

if ENCODING == "PERIODIC":
   parameters["volume"] = (2-torch.sin(grid/MAX_GRID*2*math.pi).abs()).log()
elif ENCODING == "SHIFTED":
   parameters["volume"] = (2-torch.sin(grid/MAX_GRID*2*math.pi+math.pi/2).abs()).log()
elif ENCODING == "STEEPSHIFTED":
   parameters["volume"] = (1.5-torch.sin(grid/MAX_GRID*2*math.pi+math.pi/2).abs()).log()
elif ENCODING == "UNIFORM":
   parameters["volume"] = 0*grid
elif ENCODING == "UNIMODAL2":
   parameters["volume"] = .2 * torch.cos(grid/MAX_GRID*2*math.pi+math.pi)
elif ENCODING == "UNIMODAL":
   parameters["volume"] = .5 * torch.cos(grid/MAX_GRID*2*math.pi+math.pi)
elif ENCODING == "FITTED":
   pass
elif ENCODING == "STEEPPERIODIC":
   parameters["volume"] = (1.5-torch.sin(grid/MAX_GRID*2*math.pi).abs()).log()
elif ENCODING.startswith("FOURIER"):
  _, seed = ENCODING.split("_")
  import random
  rstate = random.Random(int(seed))
  frequencies = torch.LongTensor(range(5))
  sines = torch.sin(grid.view(1,-1)*frequencies.view(-1,1) * 2*math.pi/MAX_GRID)
  cosines = torch.cos(grid.view(1,-1)*frequencies.view(-1,1) * 2*math.pi/MAX_GRID)
  basis = torch.cat([sines, cosines], dim=0)
  coefficients = torch.FloatTensor([rstate.random()-0.5 for _ in range(10)])  #/ torch.cat([frequencies, frequencies], dim=0).clamp(min=1)
  parameters["volume"] = (basis*coefficients.unsqueeze(1)).sum(dim=0)
#  figure, axis = plt.subplots(1, 1)
#  axis.scatter(grid.detach(), torch.softmax(parameters["prior"].detach(), dim=0))
#  axis.scatter(grid.detach(), torch.softmax(parameters["volume"].detach(), dim=0))
#  axis.scatter(grid.detach(), 0*grid.detach())
#  plt.show()
#  plt.close()
else:
   assert False



#assert NoiseLevels != "12345"

noiseLevelsByTrial = [int(NoiseLevels[i%len(NoiseLevels)]) for i in range(duration.size()[0])]
random.Random(11).shuffle(noiseLevelsByTrial)
duration = torch.LongTensor(noiseLevelsByTrial)

volume = 2 * math.pi * torch.nn.functional.softmax(parameters["volume"])
prior = torch.nn.functional.softmax(parameters["prior"])

observations_y = MakeZeros(xValues.size())
conditions = MakeZeros(xValues.size())
for condition in list([int(x) for x in NoiseLevels]):
   assert condition > 0 # should NOT be zero
   MASK = duration == condition
   observations_y[MASK] = samplePredictions(xValues[MASK], parameters["sigma_logit"][condition], prior, volume, n_samples=1000, grid=grid, responses_=observations_y, parameters=parameters, computePredictions=False, subject=None, sigma_stimulus=0, sigma2_stimulus=0, duration_=condition)
observations_y = observations_y.float()

SELFID = random.randint(10000, 1000000)

print(duration)
print(xValues)

with open(f"logs/SIMULATED_REPLICATE/{__file__}_{GRID}_{P}_{NoiseLevels}_N{dataSize}_{PRIOR}_{ENCODING}.txt", "w") as outFile:
    for z, y in parameters.items():
        print(z, "\t", y.detach().cpu().numpy().tolist(), file=outFile)
    print("=======", file=outFile)
    for i in range(xValues.size()[0]):
      print(int(duration[i]), round(float(grid[xValues[i]]),1), round(float(observations_y[i]),1), file=outFile)
