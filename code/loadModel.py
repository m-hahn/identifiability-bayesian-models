import json
import torch
from util import savePlot
from util import MakeZeros
from util import MakeFloatTensor
from util import MakeLongTensor


def loadModel(FILE, init_parameters, tolerant_for_sensory_noise_levels_mismatch=False):
 with open(FILE, "r") as inFile:
     (next(inFile))
     (next(inFile))
     for l in inFile:
         if l.startswith("==="):
             break
         z, y = l.split("\t")
         if tolerant_for_sensory_noise_levels_mismatch and  z.strip() == "sigma_logit":
               pass
         else:
            assert init_parameters[z.strip()].size() == MakeFloatTensor(json.loads(y)).size(), (init_parameters[z.strip()].size(), MakeFloatTensor(json.loads(y)).size())
         init_parameters[z.strip()] = MakeFloatTensor(json.loads(y))
