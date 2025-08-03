__file__ = __file__.split("/")[-1]
import math
import sys
model = sys.argv[1]
import torch
from util import MakeFloatTensor, MakeZeros
GRID = 180
from loadModel import loadModel
##############################################
# Initialize the model
# Part: Initialize the model
init_parameters = {}
init_parameters["sigma2_stimulus"] = MakeFloatTensor([0]).view(1)
init_parameters["log_motor_var"] = MakeFloatTensor([0]).view(1)
init_parameters["sigma_logit"] = MakeFloatTensor(6*[-3]).view(6)
init_parameters["mixture_logit"] = MakeFloatTensor([-1]).view(1)
init_parameters["prior"] = MakeZeros(GRID)
init_parameters["volume"] = MakeZeros(GRID)
FILE = f"logs/CROSSVALID/RunGardelle_FreePrior_CosineLoss.py_8_0_10.0_180.txt"
loadModel(FILE, init_parameters)
assert "volume" in init_parameters


def mean(x):
   return sum(x) / len(x)

import matplotlib.pyplot as plt

# Tell Matplotlib to use sans-serif by default…
plt.rcParams['font.family'] = 'sans-serif'
# …and specify Helvetica as the preferred sans-serif
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica'
plt.rcParams['mathtext.it'] = 'Helvetica:italic'
plt.rcParams['mathtext.bf'] = 'Helvetica:bold'
########################

from util import savePlot
import glob
files = glob.glob("output/evaluateCrossValidationResults_Synthetic_Gardelle_NonF.py_SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid*.py_*.txt.tex")

figures = {}
for p in [0,1,2,4,6,8]:
  figures[p] = plt.subplots(4, 3, figsize=(8,10))

def numLevelsToCol(i):
  if i == 3:
    return None
  if i == 4:
     return 2
  return i-1

dataSizes = ["N2000","N5000","N10000","N40000"]
noiseLevelsLabels = ["1 noise level","2 noise levels","4 noise levels"]

cmap = plt.cm.get_cmap("plasma", 44)

for f_ in files:
  print(f_)
 # if "ZeroTrig" not in f_:
 #     continue
  if "UNIFORM_STEEPPERIODIC" not in f_ and "STEEPPERIODIC_STEEPPERIODIC" not in f_ and "STEEPSHIFTED_STEEPPERIODIC" not in f_ and "STEEPPERIODIC_UNIFORM" not in f_:
      continue
  if "_"+model+".txt" not in f_:
    continue
  if "VarySize" not in f_:
      continue
#  if "_N5000_" not in f_:
 #     continue
  
  print(f_)
  f = f_.split(".py_")
  ps = f[2].split("_")
  print(ps)
  assert ps[0].startswith("180")
  print(ps)
  if "VarySize" not in f_:
      continue
  dataSize = ps[3]
  assert dataSize.startswith("N")
  ps = ps[1:]
  groundTruthExp = int(ps[0])



#  if groundTruthExp != 0:
 #     continue



  sensoryLevels = ps[1]
  if sensoryLevels == "56789":
    continue
  if "1" in sensoryLevels:
    continue
  if "WithStim" in f[1]:
    assert False
    stimLevels = ps[2]
  else:
    stimLevels = "0"
  print("NOISE", sensoryLevels, stimLevels, f_)
  with open(f_, "r") as inFile:
    print("READ", f_)
    data = [x.split(" ") for x in inFile.read().strip().split("\n")]
    try:
       data = [[float(q) for q in x] for x in data if x[1] != "nan" and len(x) > 2]
    except IndexError:
        print("INDEX ERROR", f_)
        continue
    data = [x for x in data if x[0] < 10]
  print("DATA", len(data))
  if len(data) == 0:
   continue
  print(f_)
  try:
     print("BEST", min([x[2] for x in data]), min([x[2] for x in data[:groundTruthExp//2-1] + data[groundTruthExp//2:]]), max([x[2] for x in data]), f_)
  except ValueError:
      print("VALUE ERROR", f_)
      continue
  
  if dataSize not in dataSizes:
      continue

  dataSizeIndex = {"N2000": 0, "N5000": 1, "N10000": 2, "N20000": None, "N40000" : 3}[dataSize]
  print("DATA SIZE INDEX", dataSizeIndex, dataSize)
  if dataSizeIndex is None:
    continue
  #if dataSizeIndex > 2:
  #    continue
  print(stimLevels)
  if stimLevels == "0":

    if sensoryLevels[0] not in ["2", "3", "4", "5"]:
       continue
    if sensoryLevels[-1] not in ["2", "3", "4", "5"]:
       continue


    # In the plot, color each line based on sensoryLevels[0]
#    colorMap = {"0" : "black", "1": "red", "2": "blue", "3": "green", "4": "purple", "5": "orange", "6" : "black", "7" : "brown", "8" : "pink", "9" : "gray"}

    # A is 1/(5 choose len(sensoryLevels))
    A = 0.5 #math.comb(5, len(sensoryLevels))**-1
    if len(sensoryLevels) != 2:
      colorHere = "gray"
    else:
      colorHere = cmap(40-10*(int(sensoryLevels[-1])-int(sensoryLevels[0])))
      colorHere = cmap(40-20*(int(sensoryLevels[-1])-int(sensoryLevels[0])-1))
 #     FIByLevel = [None, 15, 17, 20, 28, 29]
#      colorHere = cmap(40-40/11*(FIByLevel[int(sensoryLevels[-1])]-FIByLevel[int(sensoryLevels[0])]-1))
    # map 1, 2, 3 to [0,40]
 #   print("COL", 10*(mean([int(w) for w in sensoryLevels])))
#    colorHere = cmap(round(10*(mean([int(w) for w in sensoryLevels]))))

    levels_ = [int(r) for r in sensoryLevels]
    levelInLogTime = lambda x : 1/torch.sigmoid(init_parameters["sigma_logit"])[x].item()
    # TODO need to derive the overall FI from this

    if False:    
      colorHere = cmap(int(round(40-0.5*round(levelInLogTime(max(levels_))-levelInLogTime(min(levels_))))))
    else:
      colorHere = "gray"



    colIndex = numLevelsToCol(len(sensoryLevels))
    if colIndex is None:
       continue
    figures[groundTruthExp][1][dataSizeIndex, colIndex].plot([x[0] for x in data], [x[2] for x in data], color=colorHere, alpha=A) #color=colorMap[str(int(sensoryLevels[-1])-int(sensoryLevels[0]))], label=f_)
    print("Plotting data", groundTruthExp, data, f_)
    # figures[groundTruthExp][1][dataSizeIndex, len(sensoryLevels)-1].plot([0,1], [0,1])




for p in figures:
  fig, ax = figures[p]
  #for i in range(5):
 #  for j in range(5):
#    ax[j, i].set_ylim(-10,40)

  # x and y ticks labels only for leftmost and bottommost
  for i in range(3):
    for j in range(4):
      ax[j,i].set_ylim(-10,40)

      ax[j,i].tick_params(axis='both', which='major', labelsize=16)

      if i > 0:
        ax[j,i].set_yticklabels([])
      if j == 0:
        ax[j,i].set_title(noiseLevelsLabels[i], fontsize=20)
      if i == 0:
        ax[j,i].set_ylabel(dataSizes[j][1:], fontsize=20)  
      if j < 3:
        ax[j,i].set_xticklabels([])
      else:
        # labels at 0, 1, 2, 4, 6, 8
        ax[j,i].set_xticks([0,1,2,4,6,8])
      ax[j,i].set_xlim(-1,9)
      ax[j,i].plot([p,p], [-10, 40], linestyle="dotted", color="gray")
      ax[j,i].spines['top'].set_visible(False)
      ax[j,i].spines['right'].set_visible(False)
 # plt.show()
  print(f"figures/{__file__}_{model}_{p}.pdf")
  fig.savefig(f"figures/{__file__}_{model}_{p}.pdf")
  plt.close(fig)
