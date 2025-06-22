import os
import random
import subprocess
import sys

import math
import sys

import matplotlib.pyplot as plt
import glob
import random
files = os.listdir("logs/SIMULATED_REPLICATE")
random.shuffle(files)

script = f"RunSynthetic_DenseRemington_FreeEncoding_OnSim_OtherNoiseLevels_VarySize.py"

done = 0
considered = 0

for f_ in files:
  if "SimulateSynthetic2_DenseRemington_OtherNoiseLevels_VarySize.py" not in f_:
      continue
  print(f_)
  

  FIT = f_
  fold = 0
  for P in [2,4,6,8]:
    print(f"losses/{script}_{FIT}_{P}_{fold}_{10.0}_{400}.txt.txt")
    considered+=1
    if os.path.exists(f"losses/{script}_{FIT}_{P}_{fold}_{10.0}_{400}.txt.txt"):
       done+=1
       continue
    print(done/considered)

