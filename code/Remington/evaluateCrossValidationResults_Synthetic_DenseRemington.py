import sys
FIT = sys.argv[1]
if len(sys.argv) > 2 and sys.argv[2] == "--NOPLOT":
   PLOT = False
else:
   PLOT = True
__file__ = __file__.split("/")[-1]

import math
import matplotlib.pyplot as plt
from evaluateCrossValidationResults2 import crossValidResults
from matplotlib import rc
from util import savePlot
#rc('font', **{'family':'FreeSans'})

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
    if str(result[2]) == "nan":
        return
    curves[(color, style)].append((loss, result[2], result[3]))

curvesRelative = {}
def plotRelative(color, style, loss, result, resultRef):
    if (color,style) not in curvesRelative:
       curvesRelative[(color, style)] = []
    if str(result[2]) == "nan":
      return 
    if result[2] != result[2]:
        return
    sd = deltaSD(result[4],resultRef[4])
    if sd == '--':
        return
    curvesRelative[(color, style)].append((loss, result[2]-resultRef[2], sd))

curvesRelativeLF = {}
def plotEffectOfLossFunction(color, style, loss, result, reference):
    if result is None:
        return
    if (color,style) not in curvesRelativeLF:
       curvesRelativeLF[(color, style)] = []
    meanRelative = result[2] - reference[2]
    sd = deltaSD(result[4], reference[4])
    if str(result[2]) == "nan":
       return
    print("PLOTTING", result)
    #if sd == '--':
    #    return
    curvesRelativeLF[(color, style)].append((loss, meanRelative, sd))


import glob
if len(glob.glob(f"losses/Interval/*{FIT}*")) == 0:
    print("No logs exist yet", FIT)
    quit()

with open(f"output/{__file__}_{FIT}.tex", "w") as outFile:
 for loss in [0,1,2,4,6,8]:
    if loss == 0:
       full_freeprior = crossValidResults(f"Interval/RunSynthetic_DenseRemington_FreeEncoding_Zero_OnSim_OtherNoiseLevels_VarySize_Round2.py_{FIT}_{loss}_*_10.0_400.txt", STRICT=False)
    elif loss == 1:
       full_freeprior = crossValidResults(f"Interval/RunSynthetic_DenseRemington_FreeEncoding_L1_OnSim_OtherNoiseLevels_VarySize_Round2.py_{FIT}_{loss}_*_10.0_400.txt", STRICT=False)
    else:
       full_freeprior = crossValidResults(f"Interval/RunSynthetic_DenseRemington_FreeEncoding_OnSim_OtherNoiseLevels_VarySize.py_{FIT}_{loss}_*_10.0_400.txt", STRICT=False)
    cosine_freeprior = full_freeprior

    COLOR_FREE = "green"

    plot(COLOR_FREE, "solid", loss, full_freeprior)
    plot(COLOR_FREE, "dotted", loss, cosine_freeprior if loss > 0 else full_freeprior)


    RELEVANT_EXP = int(FIT.split(".py_")[1].split("_")[0 if "Dense" not in FIT else 1])
    print("EXPONENT", int(RELEVANT_EXP))

    print(f"RunSynthetic_DenseRemington_FreeEncoding_OnSim_OtherNoiseLevels_VarySize.py_{FIT}_{loss}_*_10.0_400.txt")
    print(f"RunSynthetic_DenseRemington_FreeEncoding_OnSim_OtherNoiseLevels_VarySize.py_{FIT}_{RELEVANT_EXP}_*_10.0_400.txt")

    if RELEVANT_EXP == 0:
       reference = crossValidResults(f"Interval/RunSynthetic_DenseRemington_FreeEncoding_Zero_OnSim_OtherNoiseLevels_VarySize_Round2.py_{FIT}_{RELEVANT_EXP}_*_10.0_400.txt", STRICT=False)
    elif RELEVANT_EXP == 1:
       reference = crossValidResults(f"Interval/RunSynthetic_DenseRemington_FreeEncoding_L1_OnSim_OtherNoiseLevels_VarySize_Round2.py_{FIT}_{RELEVANT_EXP}_*_10.0_400.txt", STRICT=False)
    else:
       reference = crossValidResults(f"Interval/RunSynthetic_DenseRemington_FreeEncoding_OnSim_OtherNoiseLevels_VarySize.py_{FIT}_{RELEVANT_EXP}_*_10.0_400.txt", STRICT=False)

    print("LOSS", full_freeprior)
    print("REFERENCE", reference, f"RunSynthetic_DenseRemington_FreeEncoding_OnSim_OtherNoiseLevels_VarySize.py_{FIT}_{RELEVANT_EXP}_*_10.0_400.txt")
    plotEffectOfLossFunction(COLOR_FREE, "solid", loss, full_freeprior, reference)

    plotEffectOfLossFunction(COLOR_FREE, "dotted", loss, cosine_freeprior, reference)

    print("OUTPUT", loss, cosine_freeprior[2], cosine_freeprior[2]-reference[2], "###########")
    print(loss, cosine_freeprior[2], cosine_freeprior[2]-reference[2], file=outFile)

    plotRelative(COLOR_FREE, "solid", loss, full_freeprior, full_freeprior)
    plotRelative(COLOR_FREE, "dotted", loss, cosine_freeprior if loss > 0 else full_freeprior, cosine_freeprior if loss > 0 else full_freeprior)

#print(curvesRelativeLF)
#quit()

if not PLOT:
   quit()


minY = 100000000000000
maxY = -100000000000000
figure, axis = plt.subplots(1,1, figsize=(0.9*2,0.9*2))
figure.subplots_adjust(left=0.25, bottom=0.25)
for key, values in curvesRelativeLF.items():
    color, style = key
    if len(values) == 0:
        continue
    values = [x for x in values if str(x[1]) != "nan"]
    x, y, errors = zip(*values)
    print("FOR_PLOTTING", x, y, errors)
    color = "gray"
    axis.plot(x, y, color=color, linestyle='solid', linewidth=0.5)

    minY = min(minY, min(y))
    maxY = max(maxY, max(y))
    #(_, caps, _) = axis.errorbar(x, y, yerr=[z for z in errors], color=color, fmt='none', linewidth=0.5, capsize=2)
    #for cap in caps:
    #   cap.set_markeredgewidth(0.5)
## done plotting
print(minY, maxY)
axis.set_xlim(-1,11)
axis.set_ylim(minY-20, maxY+20)
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
axis.set_yticks(ticks=[0,50,100,150], labels=[0, "", "100", ""])
axis.set_xticks(ticks=[0,5,10])
axis.tick_params(labelsize=14, width=0.4)
savePlot(f"figures/{__file__}_{FIT}_simple.pdf")
plt.show()

#######################

minY = 100000000000000
maxY = -100000000000000
figure, axis = plt.subplots(1,1, figsize=(3,3))
plt.tight_layout()
for key, values in curves.items():
    color, style = key
    if style != "dotted":
       continue
    if len(values) == 0:
        continue
    x, y, errors = zip(*values)
    print(x, y, errors)
#    i = ["solid", "dotted"].index(style)
    axis.plot(x, y, color=color, linestyle=style)
    axis.scatter(x, y, color=color)
    minY = min(minY, min(y))
    maxY = max(maxY, max(y))
    axis.errorbar(x, y, yerr=errors, color=color)

print(minY, maxY)
axis.set_ylim(minY-10, maxY+10)
axis.set_xlim(-1, 13)
axis.plot([0,16], [minY, minY], linestyle='dotted')
savePlot(f"figures/{__file__}_{FIT}.pdf")
plt.show()

figure, axis = plt.subplots(1, 2, figsize=(6,3))
plt.tight_layout()
counter = 0
for key, values in curvesRelative.items():
    counter += 1
    color, style = key
    if len(values) == 0:
        continue
    x, y, errors = zip(*values)
    x = [z + 0.2*(counter-2) for z in x]
    print(x, y, errors)
    i = ["solid", "dotted"].index(style)
    axis[i].plot(x, y, color=color, linestyle=style)
    axis[i].scatter(x, y, color=color)
    axis[i].errorbar(x, y, yerr=errors, color=color)

for i in range(2):
 axis[i].plot([0,16], [0,0])
 axis[i].set_xlabel("Exponent")
 axis[i].set_ylabel("Delta NLL")
axis[0].set_xlim(-1, 15)
axis[1].set_xlim(-1, 15)
savePlot(f"figures/{__file__}_{FIT}_Relative.pdf")
plt.show()

minY = 100000000000000
maxY = -100000000000000
figure, axis = plt.subplots(1, 2, figsize=(6,3), layout='constrained')

with open(f"output/{__file__}.txt", "w") as outFile:
 for key, values in curvesRelativeLF.items():
    color, style = key
    if len(values) == 0:
        continue
    values = [x for x in values if str(x[1]) != "nan"]
    x, y, errors = zip(*values)
    print(x, y, errors)
    i = ["solid", "dotted"].index(style)
    minY = min(minY, min(y))
    maxY = max(maxY, max(y))
    print("PLOTTING REL", x, y)
    axis[i].plot(x, y, color=color, linestyle=style)
    axis[i].scatter(x, y, color=color)
    #axis[i].errorbar(x, y, yerr=errors, color=color)
    print(color, style, [round(q) if str(q) != "nan" else "nan" for q in y], file=outFile)
    print(color, style, [round(q) if str(q) != "nan" else "nan" for q in y])

for i in range(2):
 axis[i].plot([0,16], [0,0], color="gray", linestyle="dotted")
 axis[i].set_xlabel("Exponent")
axis[0].set_ylabel("Δ NLL")
axis[0].set_ylim(minY-10, maxY+10)
axis[1].set_ylim(minY-10, maxY+10)
axis[0].set_xlim(-1, 15)
axis[1].set_xlim(-1, 15)
axis[0].set_title("Centered Loss")
axis[1].set_title("Cosine Loss")
axis[0].set_xticks(ticks=[0,2,4,6,8,10,12])
axis[1].set_xticks(ticks=[0,2,4,6,8,10,12])
savePlot(f"figures/{__file__}_{FIT}_RelativeLF.pdf")
plt.show()
