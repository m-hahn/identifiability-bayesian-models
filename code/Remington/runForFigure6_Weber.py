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

SCRIPTS = {}
SCRIPTS[0] = "RunSynthetic_DenseRemington_FreeEncoding_Zero_OnSim_OtherNoiseLevels_VarySize_Round2.py"
SCRIPTS[1] = "RunSynthetic_DenseRemington_FreeEncoding_L1_OnSim_OtherNoiseLevels_VarySize_Round2.py"
for p in [2,4,6,8,10]:
    SCRIPTS[p] = "RunSynthetic_DenseRemington_FreeEncoding_OnSim_OtherNoiseLevels_VarySize_Round2.py"

for f_ in files:
  if "SimulateRemington_Lognormal_OtherNoiseLevels_Zero" not in f_:
      continue
#  if "WEBER" not in f_:
#        continue
#  if "4567" not in f_:
#        continue
##  if "N40000" not in f_:
# #       continue
#  print(f_)
#  if "LEFT" not in f_ and "FIT" not in f_:
#        continue
#  if "Range" not in f_ and "Prior" not in f_:
#      continue
#  if "PriorSample" not in f_:
#        continue
 # if "SimulateSynthetic2_DenseRemington_OtherNoiseLevels_VarySize.py" not in f_ and "SimulateSynthetic2_OnlyRange_DenseRemington_OtherNoiseLevels_VarySize.py" not in f_ :
  #    continue
  print("Now running", f_)
  

  FIT = f_
  fold = 0
  exponents = [0,1,2,4,6,8]
  random.shuffle(exponents)
  for P in exponents:
    script = SCRIPTS[P]
    REG = "10.0"
    GRID = "400"

    REG = "0.1"
    GRID = "200"
    print(f"logs/CROSSVALID/{script}_{FIT}_{P}_{fold}_{REG}_{GRID}.txt")

    if os.path.exists(f"logs/CROSSVALID/{script}_{FIT}_{P}_{fold}_{REG}_{GRID}.txt"):
       continue
    subprocess.call([str(q) for q in ["python3", script, P, fold, REG, GRID, FIT]])
