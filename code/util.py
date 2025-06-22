import os
import math
import torch
import matplotlib.pyplot as plt
DEVICE = os.getenv('BIAS_MODEL_DEVICE')
if DEVICE not in ["cuda", "cpu"]:
   DEVICE = "cpu"
   print("Not using GPU.")
SHOW_PLOT = False

def savePlot(*args, **kwargs):
    if "transparent" not in kwargs:
      kwargs["transparent"] = True
#    kwargs["optimize"]=True
    plt.savefig(*args, **kwargs)
    if SHOW_PLOT:
       plt.show()
    plt.close()



def MakeZeros(x, *y):
    if DEVICE == 'cpu':
      return torch.zeros(x, *y)
    else:
      return torch.zeros(x, *y).cuda()
def MakeFloatTensor(x):
    if DEVICE == 'cpu':
        return torch.FloatTensor(x)
    else:
        return torch.cuda.FloatTensor(x)
def MakeLongTensor(x):
    if DEVICE == 'cpu':
        return torch.LongTensor(x)
    else:
        return torch.cuda.LongTensor(x)

def ToDevice(x):
    if DEVICE == 'cpu':
        return x.cpu()
    else:
        return x.cuda()





def toFactor(x):
    levels = sorted(list(set(x)))
    return [levels.index(z) for z in x]

def mean(x):
    return sum(x) / len(x)
def computeCenteredMean(responses_, reference):
     x1 = responses_
     x2 = responses_+360
     x3 = responses_-360
     d1 = (x1-reference).abs()
     d2 = (x2-reference).abs()
     d3 = (x3-reference).abs()
     M = torch.where(torch.logical_and(d1<d2, d1<d3), x1, torch.where(d2<d3, x2, x3)).mean()
     return M

def computeCircularSD(responses_):
      responses_ = responses_.unsqueeze(1)
      R = len(responses_)
      if R == 0:
          return float('nan')
      weights = MakeZeros(R, 1)+1/R
      angles = responses_
      responses_ = torch.stack([torch.cos(responses_/180*math.pi), torch.sin(responses_/180*math.pi)], dim=0)
      averaged = (responses_ * weights.unsqueeze(0)).sum(dim=1)
      resultantLength = averaged.pow(2).sum(dim=0).sqrt()
      circularSD = torch.sqrt(-2*torch.log(resultantLength)) * 180 / math.pi
      return float(circularSD)

def computeCircularSDWeighted(responses_, weights=None):
      angles = responses_
      responses_ = torch.stack([torch.cos(responses_/180*math.pi), torch.sin(responses_/180*math.pi)], dim=0)
      if weights is None:
         averaged = (responses_).mean(dim=1)
      else:
         averaged = (responses_ * weights.unsqueeze(0)).sum(dim=1)
      resultantLength = averaged.pow(2).sum(dim=0).sqrt()
      circularSD = torch.sqrt(-2*torch.log(resultantLength)) * 180 / math.pi
      return circularSD

def computeCircularMeanWeighted(responses_, weights=None):
      angles = responses_
      responses_ = torch.stack([torch.cos(responses_/180*math.pi), torch.sin(responses_/180*math.pi)], dim=0)
      if weights is None:
         averaged = (responses_).mean(dim=1)
      else:
         averaged = (responses_ * weights.unsqueeze(0)).sum(dim=1)
      averaged = averaged / averaged.pow(2).sum(dim=0).sqrt()
      acosine = torch.acos(averaged[0])/math.pi*180
      asine = torch.asin(averaged[1])/math.pi*180
      M = torch.where(averaged[1] > 0, acosine, 360-acosine)
      return M




def computeCircularMean(responses_):
      if sum(responses_.size()) == 0:
          return float('nan')
      angles = responses_
      responses_ = torch.stack([torch.cos(responses_/180*math.pi), torch.sin(responses_/180*math.pi)], dim=0)
      averaged = responses_.mean(dim=1)
      averaged = averaged / averaged.pow(2).sum().sqrt()
      acosine = torch.acos(averaged[0])/math.pi*180
      asine = torch.asin(averaged[1])/math.pi*180
      if averaged[0] >= 0 and averaged[1] >= 0:
          M = float(acosine)
      elif averaged[0] <= 0 and averaged[1] >= 0:
          M = float(acosine)
      elif averaged[0] <= 0 and averaged[1] <= 0:
          assert float(180-float(asine) - (360-float(acosine))) < .1
          M = 360-float(acosine)
      elif averaged[0] >= 0 and averaged[1] <= 0:
          assert abs((360+float(asine)) - ( 360-float(acosine))) < .1
          M = 360-float(acosine)
      else:
          assert False, (averaged, responses_.mean(dim=1))
      return M

def makeGridIndicesInterval(GRID, MIN_GRID, MAX_GRID):
  grid = MakeFloatTensor([x/GRID * (MAX_GRID-MIN_GRID) for x in range(GRID)]) + MIN_GRID
  grid_indices = MakeFloatTensor([x for x in range(GRID)])
  def point(p, reference):
      assert not CIRCULAR
      return p

  grid_indices_here = MakeLongTensor([[point(y,x) for x in range(GRID)] for y in range(GRID)])

def makeGridIndicesCircular(GRID, MIN_GRID, MAX_GRID):
  grid = MakeFloatTensor([x/GRID * (MAX_GRID-MIN_GRID) for x in range(GRID)]) + MIN_GRID
  grid_indices = MakeFloatTensor([x for x in range(GRID)])
  def point(p, reference):
      p1 = p
      p2 = p+GRID
      p3 = p-GRID
      ds = [abs(reference-x) for x in [p1,p2,p3]]
      m = min(ds)
      if m == ds[0]:
          return p1
      elif m == ds[1]:
          return p2
      else:
          return p3
  grid_indices_here = MakeFloatTensor([[point(y,x) for x in range(GRID)] for y in range(GRID)])
  return grid, grid_indices_here

def sign(x):
    if x < 0:
        return -1
    elif x == 0:
        return 0
    return 1

def loadParameters(init_parameters, FILE):
 import json
 with open(FILE, "r") as inFile:
    (next(inFile))
    (next(inFile))
    for l in inFile:
        if l.startswith("="):
           break
        z, y = l.split("\t")
        assert init_parameters[z.strip()].size() == MakeFloatTensor(json.loads(y)).size()
        init_parameters[z.strip()] = MakeFloatTensor(json.loads(y))

def saveModel(init_parameters, loss, crossValidLoss, lossesBy500, crossLossesBy500, FILE):
        with open(FILE, "w") as outFile:
           print(float(loss), "CrossValid", float(crossValidLoss), "CrossValidLossesBy500", " ".join([str(q) for q in crossLossesBy500]), file=outFile)
           print(iteration, "LossesBy500", " ".join([str(q) for q in lossesBy500]), file=outFile)
           for z, y in init_parameters.items():
               print(z, "\t", y.detach().cpu().numpy().tolist(), file=outFile)

def saveModelWithGrad(init_parameters, loss, crossValidLoss, lossesBy500, crossLossesBy500, FILE, maximumGradNorm):
       with open(FILE, "w") as outFile:
           print(float(loss), "CrossValid", float(crossValidLoss), "CrossValidLossesBy500", " ".join([str(q) for q in crossLossesBy500]), file=outFile)
           print(iteration, "LossesBy500", " ".join([str(q) for q in lossesBy500]), file=outFile)
           for z, y in init_parameters.items():
               print(z, "\t", y.detach().cpu().numpy().tolist(), file=outFile)
           print("========", file=outFile)
           print("\t".join([str(q) for q in maximumGradNorm]), file=outFile)

def recordLoss(crossValidLoss, FILE):
        with open(FILE, "w") as outFile:
            print(float(crossValidLoss), file=outFile)

def regularizerInterval2D(logits):
   regularizer1 = ((logits[:,1:] - logits[:,:-1]).pow(2).sum())/(logits.size()[0]*logits.size()[1])
   return regularizer1

def redularizerCircular1d(logits):
   regularizer1 = ((logits[1:] - logits[:-1]).pow(2).sum() + (logits[0] - logits[-1]).pow(2))/(logits.size()[0])

def reduceGradientsToSigns(init_parameters):
   maximumGradNorm = []
   largestGradNorm = 0
   for w in init_parameters:
     if init_parameters[w].grad is not None:
      maximumGradNorm.append(w)
      gradNormMax = float(init_parameters[w].grad.abs().max())
      maximumGradNorm.append(gradNormMax)
      largestGradNorm = max(largestGradNorm, float(gradNormMax))
      init_parameters[w].grad.data = torch.sign(init_parameters[w].grad.data)

   print(largestGradNorm, maximumGradNorm)
   return largestGradNorm, maximumGradNorm








def product(x, y):
    for z in x:
        for q in y:
            yield (z,q)

def difference(x, y):
    return (x-y).abs()


def getInverseFisherInformation(prior_2):
   volumeElement = prior_2
   fisherInformation = volumeElement.pow(2)
   inverseFisherInformation = (1/fisherInformation)
   return inverseFisherInformation

def getVBias(inverseFisherInformation, grid):
  MIN_GRID = float(grid.min())
  MAX_GRID = float(grid.max())
  GRID = grid.size()[0]
  INVERSE_DISTANCE_BETWEEN_NEIGHBORING_GRID_POINTS = GRID/(MAX_GRID-MIN_GRID)
  VBias = 0.5 * torch.autograd.grad(inverseFisherInformation.sum(), [grid], retain_graph=True)[0].detach() / (INVERSE_DISTANCE_BETWEEN_NEIGHBORING_GRID_POINTS**3)
  return VBias
#  VBias2 = 0.5 * (inverseFisherInformation[1:] - inverseFisherInformation[:-1])
#  correlationBetweenNumericalAndSymbolicV = (VBias[1:] * VBias2).mean() / (VBias[1:].pow(2).mean().sqrt() * VBias2.pow(2).mean().sqrt())
#  print("CORRELATION", correlationBetweenNumericalAndSymbolicV)
#  if correlationBetweenNumericalAndSymbolicV == correlationBetweenNumericalAndSymbolicV:
#    assert abs(float(correlationBetweenNumericalAndSymbolicV-1)) < 0.01, correlationBetweenNumericalAndSymbolicV
#  return VBias

def getWBias(prior, inverseFisherInformation, grid):
  MIN_GRID = float(grid.min())
  MAX_GRID = float(grid.max())
  GRID = grid.size()[0]
  INVERSE_DISTANCE_BETWEEN_NEIGHBORING_GRID_POINTS = GRID/(MAX_GRID-MIN_GRID)
  WBias_prior = inverseFisherInformation * (torch.autograd.grad(prior.sum(), [grid], retain_graph=True)[0].detach() / prior) / (INVERSE_DISTANCE_BETWEEN_NEIGHBORING_GRID_POINTS**3)
  return WBias_prior
#  WBias_prior2 = (inverseFisherInformation[1:] * (prior[1:] - prior[:-1]) / prior[1:]) / (INVERSE_DISTANCE_BETWEEN_NEIGHBORING_GRID_POINTS**3)
#  correlationBetweenNumericalAndSymbolicW = (WBias_prior[1:] * WBias_prior2).mean() / (WBias_prior[1:].pow(2).mean().sqrt() * WBias_prior2.pow(2).mean().sqrt())
#  print("CORRELATION", correlationBetweenNumericalAndSymbolicW)
#  if correlationBetweenNumericalAndSymbolicW == correlationBetweenNumericalAndSymbolicW:
#    assert abs(float(correlationBetweenNumericalAndSymbolicW-1)) < 0.01
#  return WBias_prior




def derivative(q):
  qGrad = 0*q
  qGrad[:-1] = (q[1:] - q[:-1]) / UNIT
  qGrad[-1] = qGrad[-2]
  qGrad[1:] = 0.5*(qGrad[1:] + qGrad[:-1])
  return qGrad

def computeSDWeighted(responses_, weights=None):
      if weights is None:
        weights = 1/responses_.size()[0]+MakeZeros(responses_.size())
#      print((responses_.pow(2) * weights).sum(dim=0), (responses_ * weights).sum(dim=0).pow(2))
      return ((responses_.pow(2) * weights).sum(dim=0) - (responses_ * weights).sum(dim=0).pow(2)).sqrt()

def computeMeanWeighted(responses_, weights=None):
      if weights is None:
        weights = 1/responses_.size()[0]+MakeZeros(responses_.size())
      return (responses_ * weights).sum(dim=0)

def sech(x):
    return 1/torch.cosh(x)

def printWithoutLeakageParts(subfigure, x, y, leak, color):
    subfigure.plot(x, [0 for _ in y], color='gray')
    subfigure.plot(x, y, c=color)

def computeCircularConfidenceInterval(data):
   ## we define the CI as an interval
   ## - covering 95% of observations
   ## - of minimal size
   results = []
   for batch in range(data.size()[1]):
     results.append([0,360])
     sort_ = sorted(data[:,batch].numpy().tolist())
     bestWidth = 360
     for i in range(0, len(sort_)):
       # consider the interval from i on
       startsAt = sort_[i]
       endsAt = sort_[(i + round(0.95 * len(sort_))) % len(sort_)]
       width = (endsAt - startsAt) % 360
       if width < bestWidth:
          results[-1] = [startsAt, endsAt]
          bestWidth = width
#       print("Considering interval", startsAt, endsAt, "with width", width)
 #      print("overall", results[-1])
  
   return MakeFloatTensor(results)


def bringCircularBiasCloseToZero(y_set):
   y_set1 = y_set-360
   y_set2 = y_set+360
   y_set = torch.where(y_set.abs() < 180, y_set, torch.where(y_set1.abs() < 180, y_set1, y_set2))
   assert (y_set.abs() <= 180).all()
   return y_set

