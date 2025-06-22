import os
import random
import subprocess
import sys
from util import savePlot

fits = [x for x in os.listdir("logs/SIMULATED_REPLICATE") if x.startswith("SimulateSynthetic2_DenseRemington")]
random.shuffle(fits)
for f in fits:
  if ("Other" not in f):
    continue
# if "Other" not in f and "Subset" not in f:
#   continue
# if "UNIFORM_UNIFORM" in f or ("FOUR" not in f and "SQRT" not in f and "STEEP" in f and "SQUARE" not in f): # and "SHIFT" in f:
  subprocess.call([str(q) for q in ["python3", "evaluateCrossValidationResults_Synthetic_DenseRemington.py", f, "--NOPLOT"]])
