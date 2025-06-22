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
#from l1Estimator import L1Estimator
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
assert P >= 0
FOLD_HERE = int(sys.argv[2])
REG_WEIGHT = float(sys.argv[3])
GRID = int(sys.argv[4])
dataSize = int(sys.argv[5])

PRIOR = sys.argv[6] #"UNIFORM"
ENCODING = sys.argv[7] #"STEEPPERIODIC"
NoiseLevels = sys.argv[8] #"25"
condition_reference = sys.argv[9] #"9"
assert condition_reference == "9"
SPREAD = float(sys.argv[10])

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

randomGenerator = random.Random(10)
observations_x = 360 * MakeFloatTensor([randomGenerator.random() for _ in range(observations_x.size()[0])])


sample = torch.cat(multitude * [sample], dim=0)[:dataSize]
responses = torch.cat(multitude * [responses], dim=0)[:dataSize]
Subject = torch.cat(multitude * [Subject], dim=0)[:dataSize]
duration = torch.cat(multitude * [duration], dim=0)[:dataSize]


#############################################################
# Part: Partition data into folds. As described in the paper,
# this is done within each subject.
N_FOLDS = 10
assert FOLD_HERE < N_FOLDS

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

#observations_x_reference = observations_x + MakeFloatTensor([20*(randomGenerator.random()-0.5) for _ in range(observations_x.size()[0])])
#observations_x_reference = observations_x_reference % 360
#
# Project observed stimuli onto grid
xValues = []
for x in observations_x:
   xValues.append(int( torch.argmin((grid - x).abs())))
xValues = MakeLongTensor(xValues)
#
#xValues_reference = []
#for x in observations_x_reference:
#   xValues_reference.append(int( torch.argmin((grid - x).abs())))
#xValues_reference = MakeLongTensor(xValues_reference)


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
elif P == 1:
  L1Estimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE)  
elif P == 0:
  # these parameters are chosen to avoid numerical problems / NaNs. No evidence that they hurt NLL.
  SCALE = 10
  KERNEL_WIDTH = 20

  MAPCircularEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, KERNEL_WIDTH=KERNEL_WIDTH, SCALE=SCALE, MIN_GRID=MIN_GRID, MAX_GRID=MAX_GRID)
else:
  assert False
#############################################################
# Part: Run the model. This function implements the model itself:
## calculating the likelihood of a given dataset under that model
## and---if the computePredictions argument is set to True--- computes
## the bias and variability of the estimate.
def samplePredictions(stimulus_, stimulus_reference_, sigma_logit, sigma_logit_reference, prior, volumeElement, n_samples=100, showLikelihood=False, grid=grid, responses_=None, parameters=None, computePredictions=False, subject=None, sigma_stimulus=None, sigma2_stimulus=None, duration_=None, folds=None):
 motor_variance = torch.exp(- parameters["log_motor_var"])
 # Part: Obtain the sensory noise variance.
 sigma2 = 4*torch.sigmoid(sigma_logit)
 sigma2_reference = 4*torch.sigmoid(sigma_logit_reference)
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
  sensory_likelihoods_reference = torch.softmax(1000000000*((SQUARED_SENSORY_SIMILARITY(F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1))))  + volumeElement.unsqueeze(1).log(), dim=0)

  if sigma2_stimulus == 0:
    likelihoods = sensory_likelihoods
    likelihoods_reference = sensory_likelihoods_reference
  else:
    assert False
    likelihoods = torch.matmul(sensory_likelihoods, stimulus_likelihoods)

  # Compute posterior
  posterior = prior.unsqueeze(1) * likelihoods.t()
  posterior = posterior / posterior.sum(dim=0, keepdim=True)

  posterior_reference = prior.unsqueeze(1) * likelihoods_reference.t()
  posterior_reference = posterior_reference / posterior_reference.sum(dim=0, keepdim=True)


  ## Compute the estimator for each m in the discretized sensory space.
  bayesianEstimate = MAX_GRID/GRID * CosineEstimator.apply(grid_indices_here, posterior)
  bayesianEstimate_reference = MAX_GRID/GRID * CosineEstimator.apply(grid_indices_here, posterior_reference)

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


  probabilityOfGivingOne_ = torch.einsum("ij,is,jt->st", comparison, density_reference, density)
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
  probabilityOfGivingOne = probabilityOfGivingOne_[stimulus_reference_, stimulus_].clamp(min=0, max=1)
  print(probabilityOfGivingOne, probabilityOfGivingOne.size(), probabilityOfGivingOne.min(), probabilityOfGivingOne.max())
  #quit()
  sampled_response = torch.bernoulli(probabilityOfGivingOne)
 # quit()
#  sampled_response = torch.where(torch.bernoulli(probabilityOfGivingOne)>0, 1, -1)
  # TODO now Bernoulli sample!

#  print(sampled_estimator.min())
 # print(sampled_estimator.max())

  return sampled_response.float(), probabilityOfGivingOne.float()

lowestError = 100000

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

#PRIOR = "STEEPPERIODIC"
#ENCODING = "UNIFORM"
#NoiseLevels = "12345"

# VISUALIZ PSYCHOMETRIC CURVE
# data = read.csv("Simulate_2AFC_Synthetic_Parameterized_OtherNoiseLevels_Grid.py_180_2_T8_3456_FOURIER_2_FOURIER_3.txt", sep=" ")
# ggplot(data %>% mutate(a=as.factor(a)) %>% mutate(delta=d-c) %>% filter(abs(delta) < 20) %>% group_by(delta, a) %>% summarise(e=mean(e)), aes(x=delta, y=e, color=a, group=a)) + geom_line()


#parameters["sigma_logit"] = MakeFloatTensor([0,-1,-3,-5,-7,-9,-11,-13,-15,-17])
#parameters["sigma_logit"][0] = -20

#[-3.0, 0.8670228123664856, -1.4027767181396484, -3.24409556388855, -3.7997844219207764, -4.360845565795898]


import counterfactualComponents 

counterfactualComponents.setPrior(PRIOR, parameters, grid, MAX_GRID)
counterfactualComponents.setEncoding(ENCODING, parameters, grid, MAX_GRID)



noiseLevelsByTrial = [int(NoiseLevels[i%len(NoiseLevels)]) for i in range(duration.size()[0])]
random.Random(11).shuffle(noiseLevelsByTrial)
duration = torch.LongTensor(noiseLevelsByTrial)

volume = 2 * math.pi * torch.nn.functional.softmax(parameters["volume"])
prior = torch.nn.functional.softmax(parameters["prior"])

volumeElement = volume

F = torch.cat([MakeZeros(1), torch.cumsum(volumeElement, dim=0)], dim=0)

observations_x_reference = 0*observations_x - 1e5

xValues = []
for x in observations_x:
   xValues.append(int( torch.argmin((grid - x).abs())))
xValues = MakeLongTensor(xValues)


for condition in list([int(x) for x in NoiseLevels]):
   assert condition > 0 # should NOT be zero
   sigma_logit = parameters["sigma_logit"][int(condition)]
   sigma_logit_reference = parameters["sigma_logit"][int(condition_reference)] 
   sigma2 = 4*torch.sigmoid(sigma_logit)
   sigma2_reference = 4*torch.sigmoid(sigma_logit_reference)
   sensory_likelihoods = torch.softmax(((SQUARED_SENSORY_SIMILARITY(F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1)))/(sigma2))  + volumeElement.unsqueeze(1).log(), dim=0)
   sensory_likelihoods_reference = torch.softmax(((SQUARED_SENSORY_SIMILARITY(F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1)))/(sigma2_reference))  + volumeElement.unsqueeze(1).log(), dim=0)
  
  # print(sensory_likelihoods[:,5])
 #  print(sensory_likelihoods_reference[:,5])
   variability = (computeCircularSDWeighted(grid.unsqueeze(1), weights=sensory_likelihoods) + computeCircularSDWeighted(grid.unsqueeze(1), weights=sensory_likelihoods_reference)) #.sqrt()
 
   MASK = duration == condition
  
#   observations_x_reference[MASK] = observations_x[MASK] + 1 * (variability[xValues[MASK]]) * MakeFloatTensor([(randomGenerator.random()-0.5) for _ in range(observations_x[MASK].size()[0])])
   observations_x_reference[MASK] = observations_x[MASK] + SPREAD * MakeFloatTensor([(randomGenerator.random()-0.5) for _ in range(observations_x[MASK].size()[0])])
 #  print(observations_x_reference[MASK] - observations_x[MASK])
#   print((variability[xValues[MASK]])/2)
 #  quit()
   observations_x_reference[MASK] = observations_x_reference[MASK] % 360
   print(condition, (observations_x_reference[MASK]-observations_x[MASK]).abs().mean(), variability.mean())



xValues_reference = []
for x in observations_x_reference:
   xValues_reference.append(int( torch.argmin((grid - x).abs())))
xValues_reference = MakeLongTensor(xValues_reference)


stimulus_ = xValues
responses_=observations_y

x_set = sorted(list(set(xValues.cpu().numpy().tolist())))



observations_y2 = MakeZeros(xValues.size())
observations_prob2 = MakeZeros(xValues.size())
observations_y = MakeZeros(xValues.size())
observations_prob = MakeZeros(xValues.size())
conditions = MakeZeros(xValues.size())
for condition in list([int(x) for x in NoiseLevels]):
   assert condition > 0 # should NOT be zero
   MASK = duration == condition
   CosineEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, P=P, SQUARED_SENSORY_DIFFERENCE=SQUARED_SENSORY_DIFFERENCE, SQUARED_SENSORY_SIMILARITY=SQUARED_SENSORY_SIMILARITY)

   observations_y[MASK], observations_prob[MASK] = samplePredictions(xValues[MASK], xValues_reference[MASK], parameters["sigma_logit"][int(condition)], parameters["sigma_logit"][int(condition_reference)], prior, volume, n_samples=1000, grid=grid, responses_=observations_y, parameters=parameters, computePredictions=False, subject=None, sigma_stimulus=0, sigma2_stimulus=0, duration_=condition)

   competingPForKLIllustration = (10-P)
   CosineEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, P=competingPForKLIllustration, SQUARED_SENSORY_DIFFERENCE=SQUARED_SENSORY_DIFFERENCE, SQUARED_SENSORY_SIMILARITY=SQUARED_SENSORY_SIMILARITY)
   observations_y2[MASK], observations_prob2[MASK] = samplePredictions(xValues[MASK], xValues_reference[MASK], parameters["sigma_logit"][int(condition)], parameters["sigma_logit"][int(condition_reference)], prior, volume, n_samples=1000, grid=grid, responses_=observations_y, parameters=parameters, computePredictions=False, subject=None, sigma_stimulus=0, sigma2_stimulus=0, duration_=condition)

observations_y = observations_y.float()

print("KL", ((observations_prob * (observations_prob.log() - observations_prob2.log()) + (1-observations_prob) * ((1-observations_prob).log() - (1-observations_prob2).log()))).sum())
print("KL emp", ((observations_y * (observations_prob.log() - observations_prob2.log()) + (1-observations_y) * ((1-observations_prob).log() - (1-observations_prob2).log()))).sum())


SELFID = random.randint(10000, 1000000)

#print(duration)
#print(xValues)

with open(f"logs/SIMULATED_REPLICATE/{__file__}_{GRID}_{P}_T{condition_reference}_S{SPREAD}_{NoiseLevels}_{dataSize}_{PRIOR}_{ENCODING}.txt", "w") as outFile:
    for z, y in parameters.items():
        print(z, "\t", y.detach().cpu().numpy().tolist(), file=outFile)
    print("=======", file=outFile)
    for i in range(xValues.size()[0]):
      print(int(duration[i]), int(condition_reference), round(float(grid[xValues[i]]),1), round(float(grid[xValues_reference[i]]),1), int(observations_y[i].item()), round(float(observations_prob[i].item()), 8), file=outFile)
