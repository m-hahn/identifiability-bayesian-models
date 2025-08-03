import os
import random
import subprocess
import sys
from util import savePlot

fits = [x for x in os.listdir("logs/SIMULATED_REPLICATE") if x.startswith("SimulateSynthetic_Parameterized")]
random.shuffle(fits)
for f in fits:
  if "NoGuess" in f:
     continue
  if "VarySize" not in f:
      continue
  if "FOURIER" in f:
      print(f)
      continue
  print("FIT", f)
  subprocess.call([str(q) for q in ["python3", "evaluateCrossValidationResults_Synthetic_Gardelle_NonF.py", f]])
