__file__ = __file__.split("/")[-1]
import math
import sys
model = sys.argv[1]

import matplotlib.pyplot as plt
from util import savePlot
import glob
files = glob.glob("output/evaluateCrossValidationResults_Synthetic_DenseRemington.py_SimulateSynthetic2_DenseRemington_OtherNoiseLevels_VarySize.py_*.txt.tex")

figures = {}
for p in [0,1,2,4,6,8]:
  figures[p] = plt.subplots(5, 4, figsize=(8,8))

dataSizes = ["N2000","N5000","N10000","N20000","N40000"]
noiseLevelsLabels = ["1 noise level","2 noise levels","3 noise levels","4 noise levels"]

cmap = plt.cm.get_cmap("plasma", 44)

for f_ in files:
 # print(f_)
  if "_"+model+".txt" not in f_:
    continue
  if "VarySize" not in f_:
      continue
  
#  print(f_)
  f = f_.split(".py_")
  ps = f[2].split("_")
  print(ps)
  assert ps[0].startswith("400")
  print(ps)
  if "VarySize" not in f_:
      continue
  dataSize = ps[3]
  assert dataSize.startswith("N")
  ps = ps[1:]
  groundTruthExp = int(ps[0])


  print(f_)

  sensoryLevels = ps[1]
#  if sensoryLevels == "56789":
#    continue
#  if "1" in sensoryLevels:
 #   continue
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

  dataSizeIndex = {"N2000": 0, "N5000": 1, "N10000": 2, "N20000": 3, "N40000" : 4}[dataSize]
  print("DATA SIZE INDEX", dataSizeIndex, dataSize)
  #if dataSizeIndex > 2:
  #    continue
  print(stimLevels)
  if stimLevels == "0":
    # In the plot, color each line based on sensoryLevels[0]
#    colorMap = {"0" : "black", "1": "red", "2": "blue", "3": "green", "4": "purple", "5": "orange", "6" : "black", "7" : "brown", "8" : "pink", "9" : "gray"}

    # A is 1/(5 choose len(sensoryLevels))
    A = 0.5 #math.comb(5, len(sensoryLevels))**-1
    print("PLOT", data)
    if len(sensoryLevels) > 4:
      continue
    if min([int(q) for q in sensoryLevels]) < 4:
      continue
    if max([int(q) for q in sensoryLevels]) > 7:
      continue
    colorHere = "gray"
#    colorHere = cmap(10*(int(sensoryLevels[-1])-int(sensoryLevels[0])))
    figures[groundTruthExp][1][dataSizeIndex, len(sensoryLevels)-1].plot([x[0] for x in data], [x[2] for x in data], color=colorHere, alpha=A) #color=colorMap[str(int(sensoryLevels[-1])-int(sensoryLevels[0]))], label=f_)
    print("Plotting data", groundTruthExp, data, f_)
    # figures[groundTruthExp][1][dataSizeIndex, len(sensoryLevels)-1].plot([0,1], [0,1])




for p in figures:
  fig, ax = figures[p]
  #for i in range(5):
 #  for j in range(5):
#    ax[j, i].set_ylim(-10,40)

  # x and y ticks labels only for leftmost and bottommost
  for i in range(4):
    for j in range(5):
      ax[j,i].set_ylim(-10,40)
      if i > 0:
        ax[j,i].set_yticklabels([])
      if j == 0:
        ax[j,i].set_title(noiseLevelsLabels[i])
      if i == 0:
        ax[j,i].set_ylabel(dataSizes[j][1:])  
      if j < 4:
        ax[j,i].set_xticklabels([])
      else:
        # labels at 0, 1, 2, 4, 6, 8
        ax[j,i].set_xticks([0,1,2,4,6,8])
      ax[j,i].set_xlim(-1,9)
      ax[j,i].plot([p,p], [-10, 40], linestyle="dotted", color="gray")

  print(f"figures/{__file__}_{model}_{p}.pdf")
  fig.savefig(f"figures/{__file__}_{model}_{p}.pdf", transparent=True)
  plt.close(fig)
