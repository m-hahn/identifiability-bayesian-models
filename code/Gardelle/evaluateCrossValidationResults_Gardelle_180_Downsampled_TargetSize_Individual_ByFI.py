__file__ = __file__.split("/")[-1]
import torch
from util import MakeFloatTensor, MakeZeros
GRID = 180
from loadModel import loadModel
from matplotlib import rc

__file__ = __file__.split("/")[-1]
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

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



import math
import matplotlib.pyplot as plt
from evaluateCrossValidationResults2 import crossValidResults
from matplotlib import rc
from util import savePlot

def mapToRow(x):
   return [0, 1, None, None, 2][x]

def mean(x):
    return sum(x)/len(x)

def round_(x):
    if str(x).lower() == "nan":
        return "--"
    else:
        return round(x)

def deltaDiff(x,y):
    if len(x) < 10 or len(y) < 10:
        return "--"
    return round(mean([x[i]-y[i] for i in range(len(x))]),1)

def deltaSD(x,y):
    if len(x) < 10 or len(y) < 10:
        return "--"
    mu = mean([x[i]-y[i] for i in range(len(x))])
    muSquared = mean([math.pow(x[i]-y[i],2) for i in range(len(x))])
    return round(math.sqrt(muSquared - math.pow(mu, 2)) / math.sqrt(10),1)

curves = {}
def plot(color, style, loss, result):
    if (color,style) not in curves:
       curves[(color, style)] = []
    if result[2] != result[2]:
        return
    curves[(color, style)].append((loss, result[2], result[3]))

curvesRelative = {}
def plotRelative(color, style, loss, result, resultRef):
    if (color,style) not in curvesRelative:
       curvesRelative[(color, style)] = []
    if result[2] != result[2]:
        return
    sd = deltaSD(result[4],resultRef[4])
    if sd == '--':
        return
    curvesRelative[(color, style)].append((loss, result[2]-resultRef[2], sd))

def plotEffectOfLossFunction(color, style, loss, result, reference):
    assert result is not None
    if result is None:
        return
    if (color,style) not in curvesRelativeLF:
       curvesRelativeLF[(color, style)] = []
    meanRelative = result[2] - reference[2]
    if meanRelative != meanRelative:
        return None
    print(result, reference, "@")
    sd = deltaSD(result[4], reference[4])
    if sd == '--':
        sd = 0
    curvesRelativeLF[(color, style)].append((loss, meanRelative, sd))

minY = 100000000000000
maxY = -100000000000000
axis = [[plt.subplots(1,1, figsize=(1,1)) for j in range(5)] for i in range(3)]
#figure, axis = plt.subplots(3, 5, figsize=(7,7))
for i in range(len(axis)):
  for j in range(len(axis[i])):
     axis[i][j][0].subplots_adjust(left=0.25, bottom=0.25)


levelSets = set()
dataSizes = set()
import glob
import re

# Get all matching file paths
file_paths = glob.glob("losses/RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize.py_2_0_10.0_180_*_*.txt.txt")

# Sets to store the matched strings
first_star_matches = set()
second_star_matches = set()

# Regex pattern to capture the two wildcard segments
pattern = re.compile(r"RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize\.py_2_0_10\.0_180_(.*?)_(.*?)\.txt\.txt")

# Extract matches
for path in file_paths:
    filename = path.split('/')[-1]  # Get the filename only
    match = pattern.match(filename)
    if match:
        first_star_matches.add(match.group(1))
        second_star_matches.add(match.group(2))

# Results
print("First * matches:", first_star_matches)
print("Second * matches:", second_star_matches)

levelSets = sorted(list(second_star_matches))
dataSizes = sorted([int(q) for q in list(first_star_matches)])


cmap = plt.cm.get_cmap("plasma", 44)

for levels in levelSets:
 for dataSize in dataSizes:
   curvesRelativeLF = {}
  
   for loss in [0,1,2,4,6,8]:
  
  
      if loss == 0:
         cosine_freeprior = crossValidResults(f"RunGardelle_FreePrior_ZeroTrig_Downsampled_TargetSize.py_{loss}_0*_10.0_180_{dataSize}_{levels}.txt", STRICT=False)
      elif loss == 1:
         cosine_freeprior = crossValidResults(f"RunGardelle_FreePrior_L1Loss_Downsampled_TargetSize.py_{loss}_0*_10.0_180_{dataSize}_{levels}.txt", STRICT=False)
      else:
         cosine_freeprior = crossValidResults(f"RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize.py_{loss}_0*_10.0_180_{dataSize}_{levels}.txt", STRICT=False)
  
      COLOR_FREE = "green"
      COLOR_UNIFORM_PRIOR = "red"
      COLOR_UNIFORM_ENCODING = "blue"
      COLOR_HARD1 = "purple"
      COLOR_HARD2 = "orange"
  
      plot(COLOR_FREE, "dotted", loss, cosine_freeprior)
      print(cosine_freeprior, "@@@")
  
      plotEffectOfLossFunction(COLOR_FREE, "dotted", loss, cosine_freeprior, crossValidResults(f"RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize.py_8_0*_10.0_180_{dataSize}_{levels}.txt", STRICT=False))
  
      plotRelative(COLOR_FREE, "dotted", loss, cosine_freeprior, cosine_freeprior)
  
  
   for key, values in curvesRelativeLF.items():
      print("DATA", key, values)
      color, style = key
      if color != COLOR_FREE or style != "dotted":
          continue
      if len(values) == 0:
          continue
      x, y, errors = zip(*values)
      print(x, y, errors)
      color = "gray"
      size = len(levels.split("-"))
      levels_ = [int(w) for w in levels.split("-")]
      print("COL", levels, max(levels_)-min(levels_))
      colorHere = "gray"
      import math
      levelInLogTime = lambda x : 1/torch.sigmoid(init_parameters["sigma_logit"])[x].item()
      # TODO need to derive the overall FI from this

#      print("LEV", [levelInLogTime(i) for i in range(6)])
#      colorHere = cmap(round(40-0.5*(max(levels_)-min(levels_))))
      colorHere = cmap(int(round(40-0.5*round(levelInLogTime(max(levels_))-levelInLogTime(min(levels_))))))
      print("COLOR", colorHere, 40-0.5*round(levelInLogTime(max(levels_))-levelInLogTime(min(levels_))))
      print("LEV",  levelInLogTime(max(levels_)), levelInLogTime(min(levels_)), 40-0.5*(levelInLogTime(max(levels_))-levelInLogTime(min(levels_))))
 #     colorHere = cmap(10*(min(levels_))) #-min(levels_)))
      if mapToRow(size-1) is None:
         continue
      axis[mapToRow(size-1)][dataSizes.index(dataSize)][1].plot(x, y, color=colorHere) #, color=color, linestyle='solid', linewidth=0.5)
  
      minY = min(minY, min(y))
      maxY = max(maxY, max(y))
      if dataSize == 1000:
        print("USE", levels, dataSize, max(y)-min(y))
  #    (_, caps, _) = axis.errorbar(x, y, yerr=[z for z in errors], color=color, fmt='none', linewidth=0.5, capsize=2)
   #   for cap in caps:
    #     cap.set_markeredgewidth(0.5)
  ## done plotting
  #print(minY, maxY)
for i in range(len(axis)):
 for j in range(len(axis[i])):
  axis[i][j][1].set_xlim(-1,11)
  maxYHere = maxY+10
  if j < 2:
   maxYHere = 80
  else:
   maxYHere = maxY+10
  axis[i][j][1].set_ylim(-4, maxYHere)
  axis[i][j][1].set_ylabel("NLL")
  axis[i][j][1].set_xlabel("Exponent")

  axis[i][j][1].spines['top'].set_visible(False)
  axis[i][j][1].spines['right'].set_visible(False)


# Save each subplot individually as a PDF
for i in range(len(axis)):
    for j in range(len(axis[i])):
        fig = axis[i][j][0]  # This is the Figure object
        filename = f"figures/{__file__}_subplot_{i}_{j}.pdf"
        fig.savefig(filename, bbox_inches='tight')


print(levelSets)
print(dataSizes)
quit()

