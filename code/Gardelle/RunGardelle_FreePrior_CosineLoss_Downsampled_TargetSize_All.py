import glob
import subprocess
import random
for _ in range(2000):
    exponent = random.choice([0,1,2,4,6,8])
    fold = 0
    SEED = random.choice(["", "-21", "-22", "-23", "-24"]) #, 25])
#    level = random.choice(["1", "2", "3", "4", "1-2", "1-3", "1-4", "2-3", "2-4", "3-4", "1-2-3", "1-3-4", "2-3-4", "1-2-3-4"])
    level = random.choice(["1", "2", "3", "4", "5", "1-2", "1-3", "1-4", "1-5", "2-3", "2-4", "2-5", "3-4", "3-5", "4-5", "1-2-3-4-5", "2-3-4-5"])
    targetSize = random.choice(["1000", "2000", "10000"]) #"3000", "4000", 
    if exponent == 0:
        script = "RunGardelle_FreePrior_ZeroTrig_Downsampled_TargetSize.py"
    elif exponent == 1:
        script = "RunGardelle_FreePrior_L1Loss_Downsampled_TargetSize.py"
    elif exponent >= 2:
        script = "RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize.py"
    else:
        assert False
    if len(level) not in [1,2]:
       continue
    levelsString = level #.replace("-", "")
    if len(glob.glob(f"losses/{script}_{exponent}_{fold}{SEED}_*_{targetSize}_{levelsString}.txt*")) > 0:
        continue
    subprocess.call(["python3", script, str(exponent), str(fold), "10.0", "180", level, targetSize, str(SEED)])
