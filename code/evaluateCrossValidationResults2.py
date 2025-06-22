import glob
import math
import sys
from util import savePlot

def crossValidResults(Regex, STRICT=True):
   files = sorted(glob.glob(f"losses/{Regex}.txt"))

   with open("../relevantLossFiles.txt", "a") as outFile:
       print("\n".join(files), file=outFile)
   FACTOR = 1

   def sd(x):
       mean = sum(x)/len(x)
       return math.sqrt( sum([y*y for y in x])/len(x) - mean*mean)
   losses = []
   CIs = []
   inSampleLosses = []
   for f in files:
     with open(f, "r") as inFile:
       loss = float(next(inFile))
       losses.append(loss*(FACTOR))
       try:
          ci = next(inFile)
       except StopIteration:
          ci = None
       CIs.append(ci)
   if STRICT and len(losses) < 10 and len(losses) > 0:
       print("Missing observations", Regex, len(losses))
       return [0, float('nan'), float("nan"), float("nan"), []]

   if len(losses) == 0:
       print("Nothing found: ", f"losses/{Regex}.txt")
       return [0, float('nan'), float("nan"), float("nan"), []]
   return [len(losses), None, sum(losses)/len(losses), sd(losses)/math.sqrt(10), losses, CIs]

if __name__=='__main__':
    Regex = sys.argv[1]
    print(crossValidResults(Regex))
