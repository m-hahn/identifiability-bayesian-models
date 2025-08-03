import os
import random
import subprocess
import sys
from util import savePlot

import math
import sys

import matplotlib.pyplot as plt
from util import savePlot
import glob
import random
files = os.listdir("logs/SIMULATED_REPLICATE")
random.shuffle(files)




for f_ in files:
  if "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_WithStimNoise.py" not in f_:
      continue
  print(f_)
 
  FIT = f_
  fold = 0
  for P in [0,1,2,4,6,8]:
    if P == 0:
       script = f"RunSynthetic_FreePrior_ZeroTrig_OnSim_WithStimNoise.py"
    elif P == 1:
       script = f"RunSynthetic_FreePrior_L1Loss_OnSim_WithStimNoise.py"
    else:
       script = f"RunSynthetic_FreePrior_CosineLoss_OnSim_WithStimNoise.py"
      
    print(f"logs/CROSSVALID/{script}_{FIT}_{P}_{fold}_{10.0}_{180}.txt")
    if os.path.exists(f"logs/CROSSVALID/{script}_{FIT}_{P}_{fold}_{10.0}_{180}.txt"):
       continue
    subprocess.call([str(q) for q in ["python3", script, P, fold, 10.0, 180, FIT]])
