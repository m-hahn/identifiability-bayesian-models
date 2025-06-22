__file__ = __file__.split("/")[-1]

import math
import matplotlib.pyplot as plt
import random
from evaluateCrossValidationResults2 import crossValidResults
from matplotlib import rc
from util import savePlot
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
    if sd == '--':
        return
    curvesRelativeLF[(color, style)].append((loss, meanRelative, sd))

with open(f"output/{__file__}.tex", "w") as outFile:
  for loss in [0,2,4,6,8]:
    if loss == 1:
        continue
    if loss == 0:
       free = crossValidResults(f"RunRemington_Free_Zero.py_{loss}_*_0.1_200.txt", STRICT=False)
    else:
       free = crossValidResults(f"RunRemington_Free.py_{loss}_*_0.1_200.txt", STRICT=False)

    COLOR_FREE = "green"
    COLOR_UNIFORM_PRIOR = "red"
    COLOR_UNIFORM_ENCODING = "blue"
    COLOR_HARD1 = "purple"
    COLOR_HARD2 = "orange"
    COLOR_HARD3 = "gray"

    plot(COLOR_FREE, "solid", loss, free)

    plotRelative(COLOR_FREE, "solid", loss, free, free)

    plotEffectOfLossFunction(COLOR_FREE, "solid", loss, free, crossValidResults(f"RunRemington_Free_Zero.py_0_*_0.1_200.txt"))

    results = []
    results.append(round_(free[2]))

    if set(results) == set(["--"]):
        continue
    results = [loss] + results

    print(" & ".join([str(q) for q in results])+"\\\\")
    print(" & ".join([str(q) for q in results])+"\\\\", file=outFile)

minY = 100000000000000
maxY = -100000000000000
figure, axis = plt.subplots(1,1, figsize=(0.9*2,0.9*2))
figure.subplots_adjust(left=0.25, bottom=0.25)
for key, values in curvesRelativeLF.items():
    color, style = key
    if color != COLOR_HARD1:
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
axis.set_yticks(ticks=[-20,-10,0,10,20], labels=["\N{MINUS SIGN}20", "", 0, "", 20])
axis.set_xticks(ticks=[0,5,10])
axis.tick_params(labelsize=14, width=0.4)

savePlot(f"figures/{__file__}_simple.pdf")
plt.show()

#######################

minY = 100000000000000
maxY = -100000000000000
figure, axis = plt.subplots(1,1, figsize=(3,3))
plt.tight_layout()
counter = 0
for key, values in curves.items():
    counter += 1
    color, style = key
    if len(values) == 0:
        continue
    x, y, errors = zip(*values)
    x = [z + 0.2*(counter-2) for z in x]
    print(x, y, errors)
    i = ["solid", "dotted"].index(style)
    axis.plot(x, y, color=color, linestyle=style)
    axis.scatter(x, y, color=color)
    minY = min(minY, min(y))
    maxY = max(maxY, max(y))
    axis.errorbar(x, y, yerr=errors, color=color)

axis.set_xlabel("Exponent")
axis.set_ylabel("NLL")

print(minY, maxY)

axis.set_xlim(-1, 11)
savePlot(f"figures/{__file__}.pdf")
plt.show()

figure, axis = plt.subplots(1, 1, figsize=(3,3))
plt.tight_layout()
counter = 0
for key, values in curvesRelative.items():
    counter += 1
    color, style = key
    if len(values) == 0:
        continue
    x, y, errors = zip(*values)
    x = [z + 0.1*(counter-2) for z in x]
    print(x, y, errors)
    i = ["solid", "dotted"].index(style)
    axis.plot(x, y, color=color, linestyle=style)
    axis.scatter(x, y, color=color)
    axis.errorbar(x, y, yerr=errors, color=color)
axis.set_xlabel("Exponent")
axis.set_ylabel("Δ NLL")
axis.set_xlim(-1, 11)
savePlot(f"figures/{__file__}_Relative.pdf")
plt.show()

minY = 100000000000000
maxY = -100000000000000
figure, axis = plt.subplots(1, 1, figsize=(2,2), layout='constrained')

counter = 0
with open(f"output/{__file__}.txt", "w") as outFile:
 for key, values in curvesRelativeLF.items():
    counter += 1
    color, style = key
    if len(values) == 0:
        continue
    x, y, errors = zip(*values)
    x = [z + 0.1*(counter-2) for z in x]
    print(x, y, errors)
    i = ["solid", "dotted"].index(style)
    minY = min(minY, min(y))
    maxY = max(maxY, max(y))
    axis.plot(x, y, color=color, linestyle=style)
    axis.scatter(x, y, color=color) #, s=10)
    #axis.errorbar(x, y, yerr=errors, color=color)
    print(color, style, [round(q) for q in y], file=outFile)

for i in range(1):
# axis.plot([0,16], [0,0], color="gray", linestyle="dotted")
 axis.set_xlabel("Exponent", fontsize=12)
axis.set_ylabel("NLL", fontsize=12)
#axis.set_ylabel("Δ NLL")
axis.set_ylim(-4, 100) #minY-10, maxY+10)
axis.set_xlim(-1, 9)
axis.set_xticks(ticks=[0,1,2,4,6,8])
axis.set_yticks(ticks=[0,50,100])
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
axis.tick_params(axis='both', which='major', labelsize=12)
savePlot(f"figures/{__file__}_RelativeLF.pdf")
plt.show()
