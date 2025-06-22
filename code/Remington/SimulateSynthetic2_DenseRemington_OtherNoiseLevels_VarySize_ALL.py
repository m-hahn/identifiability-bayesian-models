import subprocess
import sys
PRIOR = sys.argv[1] #"STEEPSHIFTED"
ENCODING = sys.argv[2] #"STEEPPERIODIC"

from itertools import combinations, chain

def power_set(iterable):
    "power_set([1, 2, 3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


for exponent in [0]: #[0,2,4,6,8]: #[2,4,6,8]:
    for dataSize in [2000, 5000, 10000, 20000, 40000]:
        for NoiseLevels_set in power_set([4, 5, 6, 7]):
#        for NoiseLevels_set in power_set([2, 3, 5, 7]):
#        for NoiseLevels_set in power_set([2, 3, 4, 8]):
#        for NoiseLevels_set in [[2,3,4,5,6,7]]:
#        for NoiseLevels_set in power_set([2, 3, 4, 5]):
#            if len(NoiseLevels_set) < 4:
 #               continue
            if len(NoiseLevels_set) == 0:
                continue
            NoiseLevels = "".join([str(w) for w in sorted(list(NoiseLevels_set))])
            subprocess.call(["python3", "SimulateSynthetic2_DenseRemington_OtherNoiseLevels_VarySize.py", str(exponent), "0", "10.0", "400", str(dataSize), PRIOR, ENCODING, NoiseLevels])

