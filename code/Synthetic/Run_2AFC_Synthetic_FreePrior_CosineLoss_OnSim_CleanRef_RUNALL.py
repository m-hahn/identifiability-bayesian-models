import glob
import subprocess
FITs = glob.glob("logs/SIMULATED_REPLICATE/Simulate_2AFC_Synthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_WithKL_CleanRef.py*")

SCRIPT = "Run_2AFC_Synthetic_FreePrior_CosineLoss_OnSim_CleanRef.py"

import random
random.shuffle(FITs)
for f in FITs:
   f = f.split("/")[-1]
  # if "T5_" not in f:
   #    continue
   if "2345" not in f:
       continue
   if "S80.0" not in f:
       continue
   if "40000" not in f:
       continue
   for P in [2,4,6,8]:
     if len(glob.glob(f"losses/{SCRIPT}*{f}_{P}_*")) > 0:
          continue
     subprocess.call(["python3", SCRIPT, str(P), "0", "10.0", "180", f])
