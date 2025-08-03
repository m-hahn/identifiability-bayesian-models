__file__ = __file__.split("/")[-1]
import math
import sys
model = sys.argv[1]

from util import savePlot
import glob
files = glob.glob("output/evaluateCrossValidationResults_Synthetic_Gardelle_NonF.py_SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid*.py_*.txt.tex")


dataSizes = ["N2000","N5000","N10000","N20000","N40000"]
noiseLevelsLabels = ["1 noise level","2 noise levels","3 noise levels","4 noise levels"]


# if the output file doesn't exist, create it
try:
  with open(f"output/{__file__}.txt", "r") as outFile:
    pass
except FileNotFoundError:
  with open(f"output/{__file__}.txt", "w") as outFile:
    print("\t".join(["dataSize", "sensoryLevels", "lowestNoise", "highestNoise", "noiseDiff", "noiseCount", "NLLSpread", "Prior", "Encoding", "P", "ConfusNLL"]), file=outFile)


prior, encoding = model.split("_")

with open(f"output/{__file__}.txt", "a") as outFile:
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

  dataSizeIndex = {"N2000": 0, "N5000": 1, "N10000": 2, "N20000": 3, "N40000" : 4}[dataSize]
  print("DATA SIZE INDEX", dataSizeIndex, dataSize)
  #if dataSizeIndex > 2:
  #    continue
  print(stimLevels)
  if stimLevels == "0":
    # In the plot, color each line based on sensoryLevels[0]
#    colorMap = {"0" : "black", "1": "red", "2": "blue", "3": "green", "4": "purple", "5": "orange", "6" : "black", "7" : "brown", "8" : "pink", "9" : "gray"}
    if sensoryLevels[0] not in ["2", "3", "4", "5"]:
       continue
    if sensoryLevels[-1] not in ["2", "3", "4", "5"]:
       continue

    # A is 1/(5 choose len(sensoryLevels))
    A = 0.5 #math.comb(5, len(sensoryLevels))**-1
    colorHere = "gray"
#    colorHere = cmap(40-10*(int(sensoryLevels[-1])-int(sensoryLevels[0])))
#    colorHere = cmap(10*(int(sensoryLevels[0])))
    print(data)
    print("\t".join([str(q) for q in [dataSize, sensoryLevels, sensoryLevels[0], sensoryLevels[-1], int(sensoryLevels[-1])-int(sensoryLevels[0]), len(sensoryLevels), max([x[2] for x in data]) - [x[2] for x in data if x[0] == groundTruthExp][0], prior, encoding, groundTruthExp ,   min([x[2] for x in data if x[0] != groundTruthExp]) - [x[2] for x in data if x[0] == groundTruthExp][0]  ]]) , file=outFile)
    print("Plotting data", groundTruthExp, data, f_)
    # figures[groundTruthExp][1][dataSizeIndex, len(sensoryLevels)-1].plot([0,1], [0,1])



