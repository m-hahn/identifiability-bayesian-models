import computations
import getObservations
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import torch
from computations import computeResources
from cosineEstimator6 import CosineEstimator
from getObservations import retrieveObservations
from loadGardelle import *
from loadModel import loadModel
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

P = sys.argv[1]
assert P in ["SECOND","BEST"] or int(P) > 1
FOLD_HERE = int(sys.argv[2])
REG_WEIGHT = float(sys.argv[3])
GRID = int(sys.argv[4])
SHOW_PLOT = False #(len(sys.argv) < 6) or (sys.argv[5] == "SHOW_PLOT")
DEVICE = 'cuda'
FIT = sys.argv[5]

if P == "BEST":
  PForStoring = P
  with open(f"output/evaluateCrossValidationResults_Synthetic_Gardelle.py_{FIT}.tex", "r") as inFile:
   data = sorted([y for y in [x.split(" ") for x in inFile.read().strip().split("\n")] if y[1] != 'nan'], key=lambda x : float(x[1]))
   print("DATA", data)
   P = int(data[0][0])
   assert P > 0
elif P == "SECOND":
  PForStoring = P
  with open(f"output/evaluateCrossValidationResults_Synthetic_Gardelle.py_{FIT}.tex", "r") as inFile:
   data = sorted([y for y in [x.split(" ") for x in inFile.read().strip().split("\n")] if y[1] != 'nan'], key=lambda x : float(x[1]))
   print("DATA", data)
   P = int(data[1][0])
   assert P > 0
else:
   P = int(P)
   PForStoring = P

#noiseConditions = "12345"
#assert "_"+noiseConditions+"_" in FIT

FILE = f"logs/CROSSVALID/{__file__.replace('_VIZ_OnlyModel', '')}_{FIT}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.txt"

if not os.path.exists(FILE):
   print("Missing model")
   assert False, FILE

# Helper Functions dependent on the device

##############################################

with open(f"logs/SIMULATED_REPLICATE/{FIT}", "r") as inFile:
  data = [z.split(" ") for z in inFile.read().strip().split("=======\n")[1].split("\n")         ]

duration__, sample__, responses__ = zip(*data)
duration__ = MakeLongTensor([int(q) for q in duration__])
duration = duration__
sample = MakeFloatTensor([float(q) for q in sample__])
responses = MakeFloatTensor([float(q) for q in responses__])
# Store observations
observations_x = sample
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

SyntheticSubjects = MakeLongTensor([i % 10 for i in range(observations_x.size()[0])])

Fold = 0*observations_x
for i in range(int(min(SyntheticSubjects)), int(max(SyntheticSubjects))+1):
    trials = [j for j in range(SyntheticSubjects.size()[0]) if SyntheticSubjects[j] == i]
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
init_parameters = {}
init_parameters["sigma2_stimulus"] = MakeFloatTensor([0]).view(1)
init_parameters["log_motor_var"] = MakeFloatTensor([0]).view(1)
init_parameters["sigma_logit"] = MakeFloatTensor(6*[-3]).view(6)
init_parameters["mixture_logit"] = MakeFloatTensor([-1]).view(1)
init_parameters["prior"] = MakeZeros(GRID)
init_parameters["volume"] = MakeZeros(GRID)
loadModel(FILE, init_parameters)
assert "volume" in init_parameters
for _, y in init_parameters.items():
    y.requires_grad = True

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
def computeBias(stimulus_, sigma_logit, prior, volumeElement, n_samples=100, showLikelihood=False, grid=grid, responses_=None, parameters=None, computePredictions=False, subject=None, sigma_stimulus=None, sigma2_stimulus=None, duration_=None, folds=None, lossReduce='mean'):

 # Part: Obtain the motor variance by exponentiating the appropriate model parameter
 motor_variance = torch.exp(- parameters["log_motor_var"])
 sigma2 = 4*torch.sigmoid(sigma_logit)
 F = torch.cat([MakeZeros(1), torch.cumsum(volumeElement, dim=0)], dim=0)

 if True:
  folds = MakeLongTensor(folds)
  if subject is not None:
    assert False
  else:
    MASK = torch.logical_and(duration==duration_, (Fold.unsqueeze(0) == folds.unsqueeze(1)).any(dim=0))
    stimulus = stimulus_[MASK]
    responses = responses_[MASK]
  assert stimulus.view(-1).size()[0] > 0

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
  bayesianEstimate = CosineEstimator.apply(grid_indices_here, posterior)

  ## Compute the motor likelihood
  ## `error' refers to the stimulus similarity between the estimator assigned to each m and
  ## the observations found in the dataset.
  ## The Gaussian or von Mises motor likelihood is obtained by exponentiating and normalizing
  error = (SQUARED_STIMULUS_SIMILARITY(360/GRID*bayesianEstimate.unsqueeze(0) - responses.unsqueeze(1)))
  log_normalizing_constant = torch.logsumexp((SQUARED_STIMULUS_SIMILARITY(grid))/motor_variance, dim=0) + math.log(2 * math.pi / GRID)
  ## The log motor likelihoods, for each pair of sensory encoding m and observed human response
  log_motor_likelihoods = (error/motor_variance) - log_normalizing_constant
  ## Obtaining the motor likelihood by exponentiating.
  motor_likelihoods = torch.exp(log_motor_likelihoods)
  ## Obtain the guessing rate, parameterized via the (inverse) logit transform as described in SI Appendix
  # Mixture of estimation and uniform response
  uniform_part = torch.sigmoid(parameters["mixture_logit"])
  ## The full likelihood then consists of a mixture of the motor likelihood calculated before, and the uniform
  ## distribution on the full space.
  motor_likelihoods = (1-uniform_part) * motor_likelihoods + (uniform_part / (2*math.pi) + 0*motor_likelihoods)

  # Now the loss is obtained by marginalizing out m from the motor likelihood
  if lossReduce == 'mean':
    loss = -torch.gather(input=torch.matmul(motor_likelihoods, likelihoods),dim=1,index=stimulus.unsqueeze(1)).squeeze(1).log().mean()
  elif lossReduce == 'sum':
    loss = -torch.gather(input=torch.matmul(motor_likelihoods, likelihoods),dim=1,index=stimulus.unsqueeze(1)).squeeze(1).log().sum()
  else:
    assert False

  ## If computePredictions==True, compute the bias and variability of the estimate
  if computePredictions:
     bayesianEstimate_byStimulus = bayesianEstimate.unsqueeze(1)/INVERSE_DISTANCE_BETWEEN_NEIGHBORING_GRID_POINTS
     bayesianEstimate_avg_byStimulus = computeCircularMeanWeighted(bayesianEstimate_byStimulus, likelihoods)
     bayesianEstimate_sd_byStimulus = computeCircularSDWeighted(bayesianEstimate_byStimulus, likelihoods)
     bayesianEstimate_sd_byStimulus = (bayesianEstimate_sd_byStimulus.pow(2) + motor_variance * math.pow(180/math.pi,2)).sqrt()

     bayesianEstimate_avg_byStimulus = torch.where((bayesianEstimate_avg_byStimulus-grid).abs()<180, bayesianEstimate_avg_byStimulus, torch.where(bayesianEstimate_avg_byStimulus > 180, bayesianEstimate_avg_byStimulus-360, bayesianEstimate_avg_byStimulus+360))
     assert float(((bayesianEstimate_avg_byStimulus-grid).abs()).max()) < 180, float(((bayesianEstimate_avg_byStimulus-grid).abs()).max())
     posteriorMaxima = grid[posterior.argmax(dim=0)]
     posteriorMaxima = computeCircularMeanWeighted(posteriorMaxima.unsqueeze(1), likelihoods)
     encodingBias = computeCircularMeanWeighted(grid.unsqueeze(1), likelihoods)
     attraction = (posteriorMaxima-encodingBias)
     attraction1 = attraction
     attraction2 = attraction+360
     attraction3 = attraction-360
     attraction = torch.where(attraction1.abs() < 180, attraction1, torch.where(attraction2.abs() < 180, attraction2, attraction3))
  else:
     bayesianEstimate_avg_byStimulus = None
     bayesianEstimate_sd_byStimulus = None
     attraction = None
 if float(loss) != float(loss):
     print("NAN!!!!")
     quit()
 return loss, bayesianEstimate_avg_byStimulus, bayesianEstimate_sd_byStimulus, attraction

## Pass data to auxiliary script used for retrieving smoothed fits from the dataset
getObservations.setData(x_set=x_set, observations_y=observations_y, xValues=xValues, duration=duration, grid=grid)

computations.setData(GRID=GRID, STIMULUS_SPACE_VOLUME=STIMULUS_SPACE_VOLUME)

def model(grid):

  lowestLoss = 10000
  for iteration in range(1):
   parameters = init_parameters
   ## In each iteration, recompute
   ## - the resource allocation (called `volume' due to a geometric interpretation)
   ## - the prior

   # Define prior and resource allocation
   volume = 2 * math.pi * torch.nn.functional.softmax(parameters["volume"], dim=0)
   prior = torch.nn.functional.softmax(parameters["prior"], dim=0)

   loss = 0

   if iteration % 100 == 0:
     gridspec = dict(width_ratios=[1,0.5,1,0,0,0,0,0,0])
     figure, axis = plt.subplots(1, 9, figsize=(.4*6,.4*3), gridspec_kw=gridspec)
     plt.tight_layout()
     figure.subplots_adjust(wspace=0.1, hspace=0.0, left=0.0, right=1)

     PRI = 0
     ENC = 2
     ATT = 4
     REP = 5
     TOT = 6
     HUM = 8
     PAD = [1,3,4,5,6,7,8]
     axis[PRI].plot(grid, prior.detach(), color="gray")
     axis[PRI].set_ylim(bottom=0)
 #    axis[ENC].set_ylim(0, 0.5)
#     axis[ENC].set_yticks(ticks=[0,0.2,0.4], labels=["0", "0.2", "0.4"])
     axis[ENC].set_ylim(0, 0.35)
     axis[ENC].set_yticks(ticks=[0,0.1,0.2,0.3], labels=["0", "", "0.2", ""])
     x_set = sorted(list(set(xValues.cpu().numpy().tolist())))

   ## Separate train and test/heldout partitions of the data
   trainFolds = [i for i in range(N_FOLDS) if i!=FOLD_HERE]
   testFolds = [FOLD_HERE]

   ## Iterate over the conditions and possibly subjects, if parameters are fitted separately.
   ## In this dataset, all parameters are fitted across subjects.
   for DURATION in range(1,10):
    if (duration == DURATION).long().sum() == 0:
       continue
    for SUBJECT in [1]:
     ## Run the model at its current parameter values.

     loss_model, bayesianEstimate_model, bayesianEstimate_sd_byStimulus_model, attraction_model = computeBias(xValues, init_parameters["sigma_logit"][DURATION], prior, volume, n_samples=1000, grid=grid, responses_=observations_y, parameters=parameters, computePredictions=(iteration%100 == 0), subject=None, sigma_stimulus=0, sigma2_stimulus=0, duration_=DURATION, folds=testFolds, lossReduce='sum')
     loss += loss_model

     if iteration % 100 == 0:
       axis[ENC].plot(grid, 2*computeResources(volume.detach(), inverse_variance=1/float((4 * torch.sigmoid(init_parameters["sigma_logit"][DURATION])))))
       y_set, sd_set = retrieveObservations(x, None, DURATION, meanMethod="circular")
       axis[TOT].plot(grid, (bayesianEstimate_model-grid).detach()/2)

     axis[ATT].plot(grid, (attraction_model).detach()/2)
     axis[REP].plot(grid, (bayesianEstimate_model-attraction_model-grid).detach()/2)

     for w in PAD:
       axis[w].set_visible(False)

     axis[PRI].set_yticks(ticks=[0], labels=[0])
     axis[REP].tick_params(labelleft=False)
     axis[TOT].tick_params(labelleft=False)

     for w in [PRI, ENC]:
       axis[w].tick_params(labelbottom=False)


     for w in range(9):
       axis[w].spines['top'].set_visible(False)
       axis[w].spines['right'].set_visible(False)

     axis[ATT].set_ylim(-25, 25)
     axis[REP].set_ylim(-25, 25)
     axis[TOT].set_ylim(-25, 25)
     axis[HUM].set_ylim(-25, 25)
     if False:
       axis[PRI].set_title("Prior")
       axis[ENC].set_title("Resources")
       axis[ATT].set_title("Attraction")
       axis[REP].set_title("Repulsion")
       axis[TOT].set_title("Bias")
       axis[HUM].set_title("Data")

   for w in [ATT, REP, TOT, HUM]:
      axis[w].set_yticks(ticks=[-20,0,20], labels=["-20°","0°","20°"])
   for w in range(9):
       axis[w].set_xticks(ticks=[0,180,360], labels=["0°", "90°", "180°"])

   if iteration % 100 == 0:
     savePlot(f"figures/{__file__}_{FIT}_{PForStoring}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.pdf")

############################3

model(grid)
