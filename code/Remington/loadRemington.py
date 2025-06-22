import torch
import glob
from util import MakeZeros

from util import MakeFloatTensor

from util import MakeLongTensor

from util import ToDevice

files = sorted(glob.glob("data/REMINGTON/Datafiles/RSG_*_100.mat"))

from scipy.io import loadmat

target = []
response = []

for f in files:
 annots = loadmat(f)
 sample = MakeFloatTensor(annots["sample"])
 responses = MakeFloatTensor(annots["response"])
 gain = MakeFloatTensor(annots["gain"])
 correct = MakeFloatTensor(annots["correct"])
 target.append(sample)
 response.append(responses)
 assert (correct-sample).abs().max() < .1, (correct-sample).abs().max()
 assert (gain-1).abs().max() < 0.1
target = torch.stack(target, dim=0)
response = torch.stack(response, dim=0)
subject = MakeFloatTensor(list(range(target.size()[0]))).view(-1, 1).expand(-1, target.size()[1])
target = target.view(-1).contiguous()
response = response.view(-1).contiguous()
Subject = subject.contiguous().view(-1).contiguous()

observations_x = target
observations_y = response

sample=target

