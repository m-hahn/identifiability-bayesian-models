import os
import random
import subprocess
import sys
from util import savePlot

fits = [x for x in os.listdir("logs/SIMULATED_REPLICATE") if x.startswith("SimulateSynthetic_Parameterized")]
random.shuffle(fits)
for f in fits:
  if "FOURIER" not in f:
    continue
#  if ("Other" not in f):
#    continue
#  if "Grid" not in f:
#    continue
#  if "BIMOD" not in f:
#     continue
#  if "UNIMODAL" in f:
#     continue
#  if not ("UNIFORM." in f):
 #   continue
# if "UNIFORM_UNIFORM" in f or ("FOUR" not in f and "SQRT" not in f and "STEEP" in f and "SQUARE" not in f): # and "SHIFT" in f:

  if "FOURIER" not in f:
    continue
  print("FIT", f)
  subprocess.call([str(q) for q in ["python3", "evaluateCrossValidationResults_Synthetic_Gardelle.py", f]])
