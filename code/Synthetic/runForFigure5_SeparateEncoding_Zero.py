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

script = f"RunSynthetic_FreePrior_ZeroTrig_OnSim_SeparateEncoding.py"



for f_ in files:
 # if "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py" not in f_ and "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_ZeroTrig.py" not in f_:
  #    continue
  if "Additive" not in f_ and "Separate" not in f_:
      continue
  print(f_)
#  if "UNIFORM_STEEPPERIODIC" not in f_ and "STEEPPERIODIC_STEEPPERIODIC" not in f_ and "STEEPSHIFTED_STEEPPERIODIC" not in f_ and "STEEPPERIODIC_UNIFORM" not in f_:
 #     continue
  if "_180_" not in f_:
      continue
  if "VarySize" not in f_:
      continue
  
  try:
      param_value, noise_level = f_.split("_180_")[-1].split("_N")[0].split("_")[:2]
  except ValueError:
      continue

  if '1' in noise_level:
      continue

  FIT = f_
  fold = 0
  for P in [0]:
    print(f"logs/CROSSVALID/{script}_{FIT}_{P}_{fold}_{10.0}_{180}.txt")
    if os.path.exists(f"logs/CROSSVALID/{script}_{FIT}_{P}_{fold}_{10.0}_{180}.txt"):
       continue
    subprocess.call([str(q) for q in ["python3", script, P, fold, 10.0, 180, FIT]])
