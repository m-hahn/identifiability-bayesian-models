import matplotlib.pyplot as plt
import math
import torch
from util import computeCircularMean, computeCenteredMean, computeCircularSD, MakeFloatTensor, computeCircularMeanWeighted, mean, computeMeanWeighted, computeCircularSDWeighted, computeCircularConfidenceInterval, bringCircularBiasCloseToZero
def setData(**kwargs):
    global x_set
    x_set = kwargs["x_set"]
    global observations_y
    observations_y=kwargs["observations_y"]
    global xValues
    xValues=kwargs["xValues"]
    global duration
    duration=kwargs["duration"]
    global grid
    grid=kwargs["grid"]
    global GRID
    GRID = grid.size()[0]
    global SQUARED_STIMULUS_SIMILARITY
    if "SQUARED_STIMULUS_SIMILARITY" in kwargs:
       SQUARED_STIMULUS_SIMILARITY=kwargs["SQUARED_STIMULUS_SIMILARITY"]
    global Subject
    if "Subject" in kwargs:
       Subject=kwargs["Subject"]

def retrieveObservations(x, Subject_, Duration_, meanMethod = "circular"):
     assert Subject_ is None
     y_set = []
     sd_set = []
     for x in x_set:
        y_here = observations_y[(torch.logical_and((xValues == x), duration==Duration_))]

        Mean1 = computeCircularMean(y_here)
        Mean2 = computeCenteredMean(y_here, grid[x])
        if abs(Mean1-grid[x]) > 180 and Mean1 > 180:
            Mean1 = Mean1-360
        elif abs(Mean1-grid[x]) > 180 and Mean1 < 180:
            Mean1 = Mean1+360
        if Duration_ > 1 and abs(Mean1-Mean2) > 180:
            print(Duration_)
            print(y_here)
            print(Mean1, Mean2, grid[x])
        if meanMethod == "circular":
            Mean = Mean1
        elif meanMethod == "centered":
            Mean = Mean2
        else:
            assert False

        bias = Mean - grid[x]
        if abs(bias) > 180:
            bias = bias+360
        y_set.append(bias)
        sd_set.append(computeCircularSD(y_here))
     return y_set, sd_set



def retrieveAndSmoothObservations(x, Duration_, meanMethod = "circular"):
     y_set = []
     sd_set = []
     for x in x_set:
        y_here = observations_y[(torch.logical_and((xValues == x), duration==Duration_))]
        Mean1 = computeCircularMean(y_here)
        Mean2 = computeCenteredMean(y_here, grid[x])

        if abs(Mean1-grid[x]) > 180 and Mean1 > 180:
            Mean1 = Mean1-360
        elif abs(Mean1-grid[x]) > 180 and Mean1 < 180:
            Mean1 = Mean1+360

        if Duration_ > 1 and abs(Mean1-Mean2) > 180:
            print(Duration_)
            print(y_here)
            print(Mean1, Mean2, grid[x])
            assert False
        if meanMethod == "circular":
            Mean = Mean1
        elif meanMethod == "centered":
            Mean = Mean2
        else:
            assert False

        bias = Mean - grid[x]
        if abs(bias) > 180:
            bias = bias+360
        y_set.append(bias)
        sd_set.append(computeCircularSD(y_here))
     y_set = MakeFloatTensor(y_set)
     sd_set = MakeFloatTensor(sd_set)
     kernel = torch.softmax(30 * SQUARED_STIMULUS_SIMILARITY(grid.unsqueeze(0) - grid.unsqueeze(1)) - 1000 * torch.isnan(y_set).unsqueeze(0), dim=0)
     y_set[torch.isnan(y_set)] = 0
     y_set = (y_set.unsqueeze(1) * kernel).sum(dim=0)
     kernel = torch.softmax(20 * SQUARED_STIMULUS_SIMILARITY(grid.unsqueeze(0) - grid.unsqueeze(1)), dim=0)
     sd_set = (sd_set.unsqueeze(1) * kernel).sum(dim=0)
     return x_set, y_set, sd_set




def retrieveAndSmoothObservationsDirect(x, Duration_, meanMethod = "circular"):
   y_set_boot = []
   sd_set_boot = []
   for _ in range(100):
     boot = torch.randint(low=0, high=observations_y.size()[0]-1, size=observations_y.size())
     observations_y_ = observations_y[boot]
     xValues_ = xValues[boot]
     duration_ = duration[boot]
     y_sets = []
     sd_sets = []
     subjects = sorted(list(set(list(Subject.numpy().tolist()))))
     if True:
       MASK = (duration_==Duration_)
       kernel = torch.softmax(10 * SQUARED_STIMULUS_SIMILARITY(grid.unsqueeze(0) - grid[xValues_[MASK]].unsqueeze(1)), dim=0)
       y_here = observations_y_[MASK]
       y_smoothed = computeCircularMeanWeighted(y_here.unsqueeze(1), kernel)
  
  
       y_set_boot.append(y_smoothed)

       sd_set = []
       for x in x_set:
        y_here = observations_y_[(torch.logical_and((xValues_ == x), duration_==Duration_))]
        sd_set.append(computeCircularSD(y_here))
       sd_set = torch.FloatTensor(sd_set)
       kernel = torch.softmax(10 * SQUARED_STIMULUS_SIMILARITY(grid.unsqueeze(0) - grid.unsqueeze(1)), dim=0)
       sd_set = (sd_set.unsqueeze(1) * kernel).sum(dim=0)
       sd_set_boot.append(sd_set)
    
   y_set = torch.stack(y_set_boot, dim=0)
   sd_set = torch.stack(sd_set_boot, dim=0)
   if True:
     y_set_var = computeCircularConfidenceInterval(y_set) 
   else:
     y_set_var = computeCircularSDWeighted(y_set) 
   if True:
     sd_set_var = torch.stack([sd_set.quantile(q=.025, dim=0), sd_set.quantile(q=.975, dim=0)], dim=1)
   else:
     sd_set_var = (sd_set.pow(2).mean(dim=0) - sd_set.mean(dim=0).pow(2)).sqrt()

   y_set = computeCircularMeanWeighted(y_set)


   y_set = bringCircularBiasCloseToZero(y_set - grid[x_set])
   y_set_var[:,0] = bringCircularBiasCloseToZero(y_set_var[:,0] - grid[x_set])
   y_set_var[:,1] = bringCircularBiasCloseToZero(y_set_var[:,1] - grid[x_set])


   return x_set, y_set, sd_set.mean(dim=0), y_set_var, sd_set_var

  

