__file__ = __file__.split("/")[-1]
import sys
FIT = sys.argv[1]

import math
import matplotlib.pyplot as plt
from evaluateCrossValidationResults2 import crossValidResults
from matplotlib import rc
from util import savePlot
#rc('font', **{'family':'FreeSans'})
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

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
        sd = 0
    curvesRelative[(color, style)].append((loss, result[2]-resultRef[2], sd))

curvesRelativeLF = {}
def plotEffectOfLossFunction(color, style, loss, result, reference):
    if result is None:
        return
    if (color,style) not in curvesRelativeLF:
       curvesRelativeLF[(color, style)] = []
    meanRelative = result[2] - reference[2]
    sd = deltaSD(result[4], reference[4])
    if sd == '--':
        sd = 0
    curvesRelativeLF[(color, style)].append((loss, meanRelative, sd))

with open(f"output/{__file__}_{FIT}.tex", "w") as outFile:
 for loss in range(0,15): # Ignore MAP for now
    if loss == 1:
        cosine_freeprior = crossValidResults(f"RunSynthetic_FreePrior_L1Loss_OnSim.py_{FIT}_{loss}_*_10.0_180.txt", STRICT=False)
    elif loss == 0:
       cosine_freeprior = crossValidResults(f"RunSynthetic_FreePrior_ZeroTrig_OnSim.py_{FIT}_{loss}_*_10.0_180.txt", STRICT=False)
    else:
       cosine_freeprior = crossValidResults(f"RunSynthetic_FreePrior_CosineLoss_OnSim"+("_WithStimNoise" if "Stim" in FIT else "")+f".py_{FIT}_{loss}_*_10.0_180.txt", STRICT=False)
    COLOR_FREE = "green"
    COLOR_UNIFORM_PRIOR = "red"
    COLOR_UNIFORM_ENCODING = "blue"
    COLOR_HARD1 = "purple"
    COLOR_HARD2 = "orange"


    plot(COLOR_FREE, "dotted", loss, cosine_freeprior)


    RELEVANT_EXP = int(FIT.split(".py_")[1].split("_")[0 if "Grid" not in FIT else 1])

    if RELEVANT_EXP == 1:
        reference = crossValidResults(f"RunSynthetic_FreePrior_L1Loss_OnSim.py_{FIT}_{RELEVANT_EXP}_*_10.0_180.txt", STRICT=False)
    elif RELEVANT_EXP == 0:
       reference = crossValidResults(f"RunSynthetic_FreePrior_ZeroTrig_OnSim.py_{FIT}_{RELEVANT_EXP}_*_10.0_180.txt", STRICT=False)
    else:
       reference = crossValidResults(f"RunSynthetic_FreePrior_CosineLoss_OnSim"+("_WithStimNoise" if "Stim" in FIT else "")+f".py_{FIT}_{RELEVANT_EXP}_*_10.0_180.txt", STRICT=False)


#    plotEffectOfLossFunction(COLOR_HARD1, "dotted", loss, cosine, reference)
    plotEffectOfLossFunction(COLOR_FREE, "dotted", loss, cosine_freeprior, reference)
#    plotEffectOfLossFunction(COLOR_UNIFORM_PRIOR, "dotted", loss, cosine_uniprior, crossValidResults(f"RunSynthetic_FreePrior_CosineLoss_OnSim"+("_WithStimNoise" if "Stim" in FIT else "")+f"_F.py_{FIT}_{RELEVANT_EXP}_*_10.0_180.txt", STRICT=False))
#    plotEffectOfLossFunction(COLOR_UNIFORM_ENCODING, "dotted", loss, cosine_uniencoding, crossValidResults(f"RunSynthetic_FreePrior_CosineLoss_OnSim"+("_WithStimNoise" if "Stim" in FIT else "")+f"_F.py_{FIT}_{RELEVANT_EXP}_*_10.0_180.txt", STRICT=False))

    print(loss, cosine_freeprior[2], cosine_freeprior[2]-reference[2], file=outFile)
    print("RESULT", loss, cosine_freeprior[2], cosine_freeprior[2]-reference[2])
    plotRelative(COLOR_FREE, "dotted", loss, cosine_freeprior, cosine_freeprior)
#    plotRelative(COLOR_UNIFORM_PRIOR, "solid", loss, full_uniprior, full_freeprior)
#    plotRelative(COLOR_UNIFORM_PRIOR, "dotted", loss, cosine_uniprior if loss > 0 else full_uniprior, cosine_freeprior if loss > 0 else full_freeprior)
#    plotRelative(COLOR_UNIFORM_ENCODING, "solid", loss, full_uniencoding, full_freeprior)
#    plotRelative(COLOR_UNIFORM_ENCODING, "dotted", loss, cosine_uniencoding if loss > 0 else full_uniencoding, cosine_freeprior if loss > 0 else full_freeprior)

minY = 100000000000000
maxY = -100000000000000
figure, axis = plt.subplots(1,1, figsize=(0.9*2,0.9*2))
figure.subplots_adjust(left=0.25, bottom=0.25)
for key, values in curvesRelativeLF.items():
    color, style = key
    if color != COLOR_FREE or style != "dotted":
        continue
    if len(values) == 0:
        continue
    x, y, errors = zip(*values)
    print(x, y, errors)
    color = "gray"
    axis.plot(x, y, color=color, linestyle='solid', linewidth=0.5)

    minY = min(minY, min(y))
    maxY = max(maxY, max(y))
    (_, caps, _) = axis.errorbar(x, y, yerr=[z for z in errors], color=color, fmt='none', linewidth=0.5, capsize=2)
    for cap in caps:
       cap.set_markeredgewidth(0.5)
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
figure, axis = plt.subplots(1, 1, figsize=(1.3,1.3), layout='constrained')

print(curvesRelativeLF)

with open(f"output/{__file__}.txt", "w") as outFile:
 for key, values in curvesRelativeLF.items():
    print("CURVE", key, values)
    color, style = key
    values = [z for z in values if str(z[1]) != 'nan']
    if len(values) == 0:
        continue
    print("@@CURVE", key, values)
    x, y, errors = zip(*values)
    print(x, y, errors, "--CURVE")
    style = "solid" #["solid", "dotted"].index(style)
    minY = min(minY, min(y))
    maxY = max(maxY, max(y))
    axis.plot(x, y, color="gray", linestyle=style)
    axis.scatter(x, y, color="gray")
#    axis.errorbar(x, y, yerr=errors, color=color)
#    print(color, style, [round(q) for q in y], file=outFile)

axis.plot([0,16], [0,0], color="gray", linestyle="dotted")
#axis.set_xlabel("Exponent")
#axis.set_ylabel("Δ NLL")
axis.set_ylim(minY-10, maxY+10)
axis.set_ylim(minY-10, maxY+10)

ymin, ymax = plt.ylim()


axis.vlines(RELEVANT_EXP, ymin, ymax, linestyles='dotted', colors='gray')




axis.set_xlim(-1, 11)
axis.set_xlim(-1, 11)
#axis.set_title("Centered Loss")
#axis.set_title("Cosine Loss")
axis.set_xticks(ticks=[0,2,4,6,8,10])
axis.set_xticks(ticks=[0,2,4,6,8,10])
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
savePlot(f"figures/{__file__}_{FIT}_RelativeLF.pdf")
plt.show()
