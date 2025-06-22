import matplotlib.pyplot as plt
import math
import torch
from util import computeCircularMean, computeCenteredMean, computeCircularSD, computeMeanWeighted, computeSDWeighted
def setData(**kwargs):
    global x_set
    x_set = kwargs["x_set"]
    global observations_y
    observations_y=kwargs["observations_y"]
    global xValues
    xValues=kwargs["xValues"]
    global grid
    grid=kwargs["grid"]
    global Subject
    if "Subject" in kwargs:
      Subject=kwargs["Subject"]
    global SQUARED_STIMULUS_SIMILARITY
    if "SQUARED_STIMULUS_SIMILARITY" in kwargs:
      SQUARED_STIMULUS_SIMILARITY=kwargs["SQUARED_STIMULUS_SIMILARITY"]



def retrieveObservations(x, Subject_):
     y_set = []
     sd_set = []
     for x in x_set:
        if Subject_ is not None:
           y_here = observations_y[(torch.logical_and((xValues == x), Subject==Subject_))]
        else:
           y_here = observations_y[((xValues == x))]
        y_set.append(float((y_here.mean() - grid[x]).cpu()))
        sd_set.append(math.sqrt(float(y_here.pow(2).mean() - y_here.mean().pow(2))))
     return y_set, sd_set



def retrieveAndSmoothObservationsDirect(x,  meanMethod = "circular"):
   y_set_boot = []
   sd_set_boot = []
   for _ in range(100):
     boot = torch.randint(low=0, high=observations_y.size()[0]-1, size=observations_y.size())
     observations_y_ = observations_y[boot]
     xValues_ = xValues[boot]
     y_sets = []
     sd_sets = []
     subjects = sorted(list(set(list(Subject.numpy().tolist()))))
     if True:
       ## bandwidth 1/500 as a fraction of the stimulus space
       kernel = torch.softmax(500*500/2 * SQUARED_STIMULUS_SIMILARITY(grid[x_set].unsqueeze(0) - grid[xValues_].unsqueeze(1)), dim=0)
       y_here = observations_y_
  #     quit()
       y_smoothed = computeMeanWeighted(y_here.unsqueeze(1), kernel)
  
  
       y_set_boot.append(y_smoothed)

       sd_set = []
       for x in x_set:
        y_here = observations_y_[((xValues_ == x))]
        sd_set.append(computeSDWeighted(y_here))
       sd_set = torch.FloatTensor(sd_set)

       ## bandwidth 1/400 as a fraction of the stimulus space
       kernel = torch.softmax(500*500/2 * SQUARED_STIMULUS_SIMILARITY(grid[x_set].unsqueeze(0) - grid[x_set].unsqueeze(1)), dim=0)
       sd_set = (sd_set.unsqueeze(1) * kernel).sum(dim=0)
       sd_set_boot.append(sd_set)
  
    
   y_set = torch.stack(y_set_boot, dim=0)
   sd_set = torch.stack(sd_set_boot, dim=0)
   y_set_var = computeSDWeighted(y_set) #.pow(2).mean(dim=0) - y_set.mean(dim=0).pow(2)).sqrt()
#   quit()
   y_set_var = torch.stack([torch.quantile(y_set, q=0.025, dim=0), torch.quantile(y_set, q=0.975, dim=0)], dim=1)
   sd_set_var = torch.stack([torch.quantile(sd_set, q=0.025, dim=0), torch.quantile(sd_set, q=0.975, dim=0)], dim=1)
#   sd_set_var = computeSDWeighted(sd_set)

   y_set = computeMeanWeighted(y_set)
   sd_set = computeMeanWeighted(sd_set)


   #figure, axis = plt.subplots(1, 2, figsize=(8,8))
  # for i in y_set_boot:
 #    axis[0].plot(grid[x_set], i, alpha=0.1)
#   axis[0].plot(grid[x_set], y_set)

   y_set = y_set - grid[x_set]
   y_set_var = y_set_var - grid[x_set].unsqueeze(1)
#   y_set1 = y_set-360
#   y_set2 = y_set+360
#   y_set = torch.where(y_set.abs() < 180, y_set, torch.where(y_set1.abs() < 180, y_set1, y_set2))
 #  axis[1].plot(grid[x_set], y_set)
#   plt.show()


   return grid[x_set], y_set, sd_set, y_set_var, sd_set_var

 
