# Model Code

* [code for generic circular spaces](Synthetic/)
* [code for data from de Gardelle et al 2010](Gardelle/)
* [code for data from Remington et al 2018, and generic interval spaces](Remington/)

The code base is derived from that of Hahn&Wei 2024 (https://gitlab.com/m-hahn/unifying-theory-biases), and uses the same utility scripts.

# Installation
# Installation guide that includes information on the operating system, programing language, software
dependencies and non-standard hardware or resources needed to run the program and details of typical
install time on a current computer.
Demo that runs the code/software in example data and typical run time.
Provide a link to the code in an open source repository and a digital object identifier (DOI); when available.
License of use; we recommend using a license approved by the open source initiative. Please note that an open
license for code published in association with a Nature journal paper is compatible with the terms laid out in
the Nature journal License to Publish.
We strongly recommend that you ask colleagues that are not familiar with the tool to test it prior to submission.

#  System requirements
The code has been developed and tested with the following dependencies:
* Python XXX
* Software dependencies (including version numbers)
* Versions the software has been tested on>
* The code can be run on a standard computer. The fitting procedure  is sped up considerably by running on a CUDA-enabled GPU.

#  Installation guide
No installation of the code itself is needed, beyond the software dependencies listed above.
Instructions
Typical install time on a "normal" desktop computer

#  Demo
## Instructions to run on data

We use as demo dataset a [simulated dataset](logs/SIMULATED_REPLICATE/SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt) with one noise level and 1K trials.

In order to fit a model at `p=2` on this data, run:

```
cd Synthetic
rm losses/RunSynthetic_FreePrior_CosineLoss_OnSim.py_SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt_2_0_10.0_180.txt.txt
python3 RunSynthetic_FreePrior_CosineLoss_OnSim.py 2 0 10.0 180 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt
```

Expected runtime on a standard laptop or desktop computer: 5-10 minutes

Expected output:

```
more losses/RunSynthetic_FreePrior_CosineLoss_OnSim.py_SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt_2_0_10.0_180.txt.txt
```
is expected to produce `44.991607666015625` plus/minus 0.01. Small numerical deviations are possible.

#  Instructions for use

All commands are carried out in `Synthetic/`.

The code expects the de Gardelle et al 2011 dataset to be present. See 

## Simulating a dataset


```
python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py <P> 0 10.0 180 <TRIALS> <PRIOR> <ENCODING> <NOISE_LEVELS>
```

where

```
P = 1, 2, 4, 6, 8

where

```
FILE = f"logs/CROSSVALID/{__file__.replace('_VIZ', '')}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.txt"
```

In the case `P=0`, the script is instead `SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_ZeroTrig.py`, with otherwise identical arguments.


This creates a file `logs/SIMULATED_REPLICATE/SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_<P>_<NOISE_LEVELS>_N<TRIALS>_<PRIOR>_<ENCODING>.txt`.

Example:

python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 

logs/SIMULATED_REPLICATE/SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt



## Fitting models

python3 RunSynthetic_FreePrior_CosineLoss_OnSim.py 2 0 10.0 180 <NAME_OF_DATASET>

where `logs/SIMULATED_REPLICATE/<NAME_OF_DATASET>` is the path of the file stored when simulating.



