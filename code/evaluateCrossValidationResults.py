import math
import glob
import sys

def crossValidResults(Regex, STRICT=True):
   files = sorted(glob.glob(f"logs/CROSSVALID/{Regex}"))

   FACTOR = 1
   if "Undelayed" in Regex:
       FACTOR = 1/10*7561
   elif "Bae" in Regex and "BySubjEffects" in Regex:
       FACTOR = 1
   elif "BothN" in Regex or "Solomon" in Regex:
       FACTOR = 1/10*4320
   elif "Gardelle" in Regex:
      FACTOR = 1/10*15121
   elif "Xiang" in Regex:
       FACTOR = 100
   elif "Remington" in Regex:
       FACTOR = 100
   else:
       assert False, Regex
   
   def sd(x):
       mean = sum(x)/len(x)
       return math.sqrt( sum([y*y for y in x])/len(x) - mean*mean)
   losses = []
   inSampleLosses = []
   for f in files:
     with open(f, "r") as inFile:
       line = next(inFile).split(" ")
       inSample = float(line[0])
       loss = float(line[2])
       iterations = len(line)-4
#       print(iterations)
       line = next(inFile).split(" ")
       inSampleLosses = [float(x) for x in line[2:]]
       if inSample != min(inSampleLosses):
           pass
        #  print(inSample, min(inSampleLosses), line[-1])
       elif iterations < 3:
           print("REMOVING", f)
#           print("Only ", iterations, " iterations: ",f)
           continue
           pass
       losses.append(loss*(FACTOR))
       inSampleLosses.append(inSample*(9/10*FACTOR))
   if STRICT and len(losses) < 10 and len(losses) > 0:
       print("Missing observations", Regex, len(losses))
       return [0, float('nan'), float("nan"), float("nan"), []]

   if len(losses) == 0:
       print("Nothing found: ", Regex)
       return [0, float('nan'), float("nan"), float("nan"), []]
   return [len(losses), sum(inSampleLosses)/len(inSampleLosses), sum(losses)/len(losses), sd(losses)/math.sqrt(10), losses]

if __name__=='__main__':
    Regex = sys.argv[1]
    print(crossValidResults(Regex))

