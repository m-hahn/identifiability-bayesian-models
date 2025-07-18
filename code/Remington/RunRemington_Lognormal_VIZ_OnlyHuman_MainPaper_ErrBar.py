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
from getObservations import retrieveAndSmoothObservationsDirect
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
from util import sign
__file__ = __file__.split("/")[-1]

rc('font', **{'family':'FreeSans'})

# Helper Functions dependent on the device

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

LPEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, P=P, SCALE=SCALE)

#############################################################
# Part: Initialize the model
init_parameters = {}
init_parameters["sigma2_stimulus"] = MakeFloatTensor([0]).view(1)
init_parameters["log_motor_var"] = MakeFloatTensor([3]).view(1)
init_parameters["sigma_logit"] = MakeFloatTensor(1*[-3]).view(1)
init_parameters["mixture_logit"] = MakeFloatTensor([-1]).view(1)
init_parameters["prior_mu"] = MakeFloatTensor([float(observations_x.median().log())]).view(1)
init_parameters["prior_sigma2"] = MakeFloatTensor([0]).view(1)
init_parameters["prior"] = MakeZeros(GRID)
init_parameters["volume"] = MakeZeros(1)
for _, y in init_parameters.items():
    y.requires_grad = True

assert "volume" in init_parameters
for _, y in init_parameters.items():
    y.requires_grad = True

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

  # Estimator
  bayesianEstimate = LPEstimator.apply(grid[grid_indices_here], posterior)

  assert bayesianEstimate.max() <= 1.1*MAX_GRID
  assert bayesianEstimate.min() >= 0.9*MIN_GRID
  # Caculate motor likelihood

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

## Pass data to auxiliary script used for retrieving smoothed fits from the dataset
getObservations.setData(x_set=x_set, observations_y=observations_y, xValues=xValues, grid=grid, Subject=Subject, SQUARED_STIMULUS_SIMILARITY=SQUARED_STIMULUS_SIMILARITY)

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

   epsilon = torch.exp(parameters["volume"][0])
   volume = SENSORY_SPACE_VOLUME * torch.softmax(-torch.log(epsilon + grid), dim=0)
   ## For interval datasets, we're norming the size of the sensory space to 1

   prior = torch.nn.functional.softmax(-((epsilon+grid).log()-parameters["prior_mu"]).pow(2)/(1e-5+torch.exp(-parameters["prior_sigma2"])) - (epsilon+grid).log())

   loss = 0

   ## Create simple visualzation of the current fit once every 500 iterations
   if iteration % 500 == 0:

     assigned = ["PRI", None, None, "ENC", None, None, "ATT", None, "REP", None, None, "HUM", None, None, "VAH"]
     for w, k in enumerate(assigned):
         globals()[k] = w
     PAD = [w for w, q in enumerate(assigned) if q not in ["HUM", "VAH"]]
     gridspec = dict(width_ratios=[1 if x is not None else .2 for x in assigned])
     figure, axis = plt.subplots(1, len(gridspec["width_ratios"]), figsize=(14,2.1), gridspec_kw=gridspec)
     plt.tight_layout()
     figure.subplots_adjust(wspace=0.1, hspace=0.0)

     for w in PAD:
       axis[w].set_visible(False)

     for w in range(len(assigned)):
       axis[w].tick_params(axis='both', labelsize=12)
     grid_cpu = grid.detach().cpu()
     MASK = torch.logical_and(grid > float(target.min()), grid < float(target.max()))
     axis[ENC].plot(grid_cpu, computeResources(volume, torch.sigmoid(init_parameters["sigma_logit"][0])).detach().cpu(), linewidth=3)
     axis[ENC].set_ylim(0,1.1*float(computeResources(volume, torch.sigmoid(init_parameters["sigma_logit"][0])).detach().cpu().max()))
     axis[PRI].plot(grid_cpu, prior.detach().cpu(), linewidth=3)
     axis[PRI].plot([float(sample.min()), float(sample.max())], 2*[0.5*float(prior.max())], color="orange")
     x_set = sorted(list(set(xValues.cpu().numpy().tolist())))

   ## Separate train and test/heldout partitions of the data
   trainFolds = [i for i in range(N_FOLDS) if i!=FOLD_HERE]
   testFolds = [FOLD_HERE]

   ## Iterate over the conditions and possibly subjects, if parameters are fitted separately.
   ## In this dataset, there is just one condition, and all parameters are fitted across subjects.
   for DURATION in range(1):
    for SUBJECT in [1]:
     ## Run the model at its current parameter values.
     loss_model, bayesianEstimate_model, bayesianEstimate_sd_byStimulus_model, attraction = computeBias(xValues, init_parameters["sigma_logit"][DURATION], prior, volume, n_samples=1000, grid=grid, responses_=observations_y, parameters=parameters, computePredictions=(iteration%100 == 0), subject=None, sigma_stimulus=0, sigma2_stimulus=0, folds=testFolds, lossReduce='sum')
     loss += loss_model
     print("LOSS", loss)

     if iteration % 500 == 0:

       Xs, ys, sds, y_errbar, sd_errbar = retrieveAndSmoothObservationsDirect(x)

       print(len(Xs), len(ys), "LINE 362")
       axis[REP].plot(grid_cpu[MASK], (bayesianEstimate_model-grid-attraction).detach().cpu()[MASK], linewidth=3)
       axis[ATT].plot(grid_cpu[MASK], attraction[MASK].detach(), linewidth=3)
       axis[HUM].plot(Xs, ys, color="gray", linewidth=.4)

       CI_x = [float(grid[x]) for x in x_set]
       CI_y1 = ((y_errbar[:,0])).detach().numpy().tolist()
       CI_y2 = ((y_errbar[:,1])).detach().numpy().tolist()
       poly = plt.Polygon(list(zip(CI_x + CI_x[::-1], CI_y1 + CI_y2[::-1])), facecolor=(.65, .65, .65), alpha=0.5)
       axis[HUM].add_patch(poly)

       CI_x = [float(grid[x]) for x in x_set]
       CI_sd1 = ((sd_errbar[:,0])).detach().numpy().tolist()
       CI_sd2 = ((sd_errbar[:,1])).detach().numpy().tolist()
       poly = plt.Polygon(list(zip(CI_x + CI_x[::-1], CI_sd1 + CI_sd2[::-1])), facecolor=(.65, .65, .65), alpha=0.5)

       axis[VAH].add_patch(poly)

       axis[VAH].plot(Xs, sds, color="gray", linewidth=.4)

   if iteration % 500 == 0:
     axis[ENC].set_xlim(0.3, 1.2)
     axis[PRI].set_xlim(0.3, 1.2)

     if True:
        axis[ENC].set_title("Resources", fontsize=22)
        axis[PRI].set_title("Prior", fontsize=22)
        axis[REP].set_title("Repulsion", fontsize=22)
        axis[ATT].set_title("Attraction", fontsize=22)
     if True:

        axis[VAH].text(-0.1, 0.1, "[sec]", fontsize=18)
        axis[HUM].text(-0.1, 0.13, "[sec]", fontsize=18)
     axis[ENC].set_ylim(0,18)
     for z in range(len(assigned)):
         axis[z].spines['right'].set_visible(False)
         axis[z].spines['top'].set_visible(False)
         axis[z].set_xticks(ticks=[0.5, 1], labels=["0.5", "1.0"])
     for i in [REP, ATT, HUM]:
        axis[i].set_xlim(0.3, 1.2)
        axis[i].set_ylim(-0.1, 0.1)
     for i in [VAH]:
        axis[i].set_xlim(0.3, 1.2)
        axis[i].set_ylim(0,0.1)
     for z in [REP, ATT, HUM]:
         axis[z].set_yticks(ticks=[-0.1, -0.05, 0, 0.05, 0.1], labels=["-0.1", "", "0", "", "0.1"])
     for z in [VAH]:
         axis[z].set_yticks(ticks=[0, 0.02, 0.04, 0.06, 0.08, 0.1], labels=["0", "", "", "", "0.08", ""])
     for w in range(1,len(assigned)):
         if w-1 not in PAD:
             axis[w].set_yticks([])
     figure.subplots_adjust(bottom=0.2,top=0.8)
     for w in range(len(assigned)):
       axis[w].tick_params(axis='both', labelsize=18)

     for w in range(len(assigned)):
        if w in [REP]:
           axis[w].tick_params(labelleft=False)

     for w in range(len(assigned)):
       axis[w].tick_params(labelbottom=False)
     savePlot(f"figures/{__file__}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.pdf", transparent=True)

   if iteration % 500 == 0:
     from matplotlib import cm
     from matplotlib.colors import ListedColormap, LinearSegmentedColormap
     viridis = cm.get_cmap('viridis', 8)

     figure, axis = plt.subplots(1, 1, figsize=(1.5,1.5))
     axis.plot([0,0], [0, (bayesianEstimate_model-grid-attraction).detach().pow(2).mean()], linewidth=50)
     axis.plot([1,1], [0, (attraction).detach().pow(2).mean()], linewidth=50)
     plt.xticks([])
     plt.yticks([])
     savePlot(f"figures/{__file__}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}_Magnitudes.pdf")

     figure, axis = plt.subplots(1, 1, figsize=(1.8,1.8))
     plt.tight_layout()
     ratio = (prior / volume.pow((P+2)/2)).detach()
     kernel = torch.softmax(-(((grid.unsqueeze(0) - grid.unsqueeze(1))/MAX_GRID)).pow(2)/0.001, dim=0)
     ratio = torch.matmul(ratio.unsqueeze(0), kernel).view(-1)
     prior_smoothed = torch.matmul(prior.unsqueeze(0), kernel).view(-1)
     volume_smoothed = torch.matmul(volume.unsqueeze(0), kernel).view(-1)
     axis.plot([float(target.min()), float(target.max())], [8,8], color="gray", linestyle='dotted')
     axis.plot([float(target.min()), float(target.max())], [0,0], color="gray", linestyle='dotted')
     axis.plot([float(target.min()), float(target.max())], [-8,-8], color="gray", linestyle='dotted')
     axis.plot([float(target.min()), float(target.max())], [-16,-16], color="gray", linestyle='dotted')
     plt.yticks([])

     axis.set_yticks(ticks=[8, 0, -8, -16], labels=["A", "B", "C", "D"])
     axis.set_xlim(0.4, 1.2)

     for w in range(2,GRID):
        if grid[w] < float(target.min()) or grid[w] > float(target.max()):
            continue

        y1 = float(ratio[w-1]-ratio[w-2])
        y2 = float(ratio[w]-ratio[w-1])

        repulsive_part = float((bayesianEstimate_model-grid-attraction)[w].cpu().detach())
        attractive_part = float((attraction)[w].cpu().detach())
        repulsivePredominance = math.pow(repulsive_part, 2) / (math.pow(repulsive_part,2) + math.pow(attractive_part,2))

        hasAttraction = False
        repulsion = False
        if sign(float(prior_smoothed[w]-prior_smoothed[w-1])) == sign(y2):
            hasAttraction=True
        if sign(float(volume_smoothed[w]-volume_smoothed[w-1])) == -sign(y2):
            repulsion=True
        if hasAttraction and repulsion:
            c = "gray"
        elif hasAttraction:
            c = "green"
        elif repulsion:
            c = "blue"
        else:
            c = "black"

        axis.plot([float(grid[w-1]), float(grid[w])], [0+sign(y2),0+sign(y2)], color=["green", "red", "gray"][{-1 : 0, 0 : 2, 1 : 1}[sign(y2)]])

        y2 = repulsive_part
        axis.plot([float(grid[w-1]), float(grid[w])], [-8+sign(y2),-8+sign(y2)], color=["green", "red", "gray"][{-1 : 0, 0 : 2, 1 : 1}[sign(y2)]])

        y2 = attractive_part
        axis.plot([float(grid[w-1]), float(grid[w])], [-16+sign(y2),-16+sign(y2)], color=["green", "red", "gray"][{-1 : 0, 0 : 2, 1 : 1}[sign(y2)]])

     for w in range(2, Xs.size()[0]):
        y2 = ys[w]

        axis.plot([float(Xs[w-1]), float(Xs[w])], [8+sign(y2),8+sign(y2)], color=["green", "red", "gray"][{-1 : 0, 0 : 2, 1 : 1}[sign(y2)]])

     savePlot(f"figures/{__file__}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}_PPRatio.pdf")

############################3

model(grid)
