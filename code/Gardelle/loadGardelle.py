import torch
from util import MakeZeros

from util import MakeFloatTensor

from util import MakeLongTensor

from util import toFactor

with open("data/GARDELLE/data.txt", "r") as inFile:
    data = [x.split("\t") for x in inFile.read().strip().split("\n")]
    header = [x.strip('"') for x in data[0]]
    header = dict(list(zip(header, range(len(header)))))
    print(header)
    data = data[1:]
sample = 2*MakeFloatTensor([float(x[header["rot"]]) for x in data])
responses = 2*MakeFloatTensor([float(x[header["resp_rot"]]) for x in data])
duration = MakeLongTensor(toFactor([int(x[header["targdur"]]) for x in data]))
Subject = MakeFloatTensor(toFactor([x[header["sub"]] for x in data]))

mask = torch.ByteTensor([float(x[header["rot"]]) not in [135, 90, 45, 0] for x in data])
print(mask.float().sum(), (1-mask.float()).sum())

sample = sample[mask]
responses = responses[mask]
duration = duration[mask]
Subject = Subject[mask]


observations_x = sample
observations_y = responses

