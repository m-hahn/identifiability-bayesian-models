import torch
import glob
from util import MakeZeros

from util import MakeFloatTensor

from util import MakeLongTensor

from util import ToDevice


from scipy.io import loadmat

target = []
response = []

with open("data/simulated.txt", "r") as inFile:
 target, response = zip(*[x.split(" ") for x in inFile.read().strip().split("\n")])
 target = MakeFloatTensor([float(x) for x in target])
 response = MakeFloatTensor([float(x) for x in response])
## the simulated dataset does not contain a column  indicating the subject
Subject = 0 * target  

observations_x = target
observations_y = response

sample=target

