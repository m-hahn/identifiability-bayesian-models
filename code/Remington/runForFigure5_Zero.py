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

script = f"RunSynthetic_DenseRemington_FreeEncoding_Zero_OnSim_OtherNoiseLevels_VarySize_Round2.py"



for f_ in files:
  if "SimulateSynthetic2_DenseRemington_OtherNoiseLevels_VarySize.py" not in f_:
      continue
  print(f_)
  

  FIT = f_
  fold = 0
  for P in [0]:
    print(f"Checking logs/CROSSVALID/{script}_{FIT}_{P}_{fold}_{10.0}_{400}.txt")
    if os.path.exists(f"logs/CROSSVALID/{script}_{FIT}_{P}_{fold}_{10.0}_{400}.txt"):
       continue
    subprocess.call([str(q) for q in ["python3", script, P, fold, 10.0, 400, FIT]])
