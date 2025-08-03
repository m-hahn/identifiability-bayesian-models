# Code

## Overview

* [code for generic circular spaces](Synthetic/)
* [code for data from de Gardelle et al 2010](Gardelle/)
* [code for data from Remington et al 2018, and generic interval spaces](Remington/)

The code base is derived from that of Hahn&Wei 2024 (https://gitlab.com/m-hahn/unifying-theory-biases), and uses the same utility scripts.

## Getting Started

###  System requirements
The code has been developed and tested with the following dependencies:
* Python 3.9.18, with packages listed at the [list of Python packages](requirements.txt). We ran the code in a Conda virtual environment (Conda version 23.7.4).
* The code can be run on a standard computer on the command line. The fitting procedure is sped up considerably by running on a CUDA-enabled GPU. For this, set `BIAS_MODEL_DEVICE=cuda`.

###  Installation guide
No installation of the code itself is needed, beyond the software dependencies listed above.
For getting started, we recommend creating a Conda virtual environment: 

```
conda create --name identifiability python=3.9 --file requirements.txt -y
conda activate identifiability 
```

### Demo: Instructions to run on demo data

The demo is in the working directory `Synthetic/`.
We use as demo dataset a [simulated dataset](logs/SIMULATED_REPLICATE/SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt) with one noise level and 1K trials.

In order to fit a model at `p=2` on this data, first delete the existing log:

```
rm losses/RunSynthetic_FreePrior_CosineLoss_OnSim.py_SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt_2_0_10.0_180.txt.txt
```

and then run the fitting script:

```
python3 RunSynthetic_FreePrior_CosineLoss_OnSim.py 2 0 10.0 180 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt
```

Expected runtime on a standard laptop or desktop computer: 5-10 minutes

Expected output:

```
more losses/RunSynthetic_FreePrior_CosineLoss_OnSim.py_SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt_2_0_10.0_180.txt.txt
```
is expected to produce `44.991607666015625` plus/minus 0.01. Small numerical deviations are possible.

You can plot the model fit by running

```
python3 RunSynthetic_FreePrior_CosineLoss_OnSim_VIZ.py 2 0 10.0 180 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt
```

which produces this plot [here](figures/RunSynthetic_FreePrior_CosineLoss_OnSim_VIZ.py_SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt_2_0_10.0_180.pdf).

#  Instructions for use

Here, we provide instructions for use on circular stimulus spaces. All commands are carried out in `Synthetic/`.
The code expects the de Gardelle et al 2011 dataset to be present. See TODO for instructions.

## Simulating a dataset


```
python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py <P> 0 10.0 180 <TRIALS> <PRIOR> <ENCODING> <NOISE_LEVELS>
```

where

```
P = 1, 2, 4, 6, 8

TRIALS (number of trials) is any positive integer (e.g., 1000, 10000)

PRIOR and ENCODING are defined in `counterfactualComponents.py`. Relevant options include `UNIFORM`, `STEEPPERIODIC`, `STEEPSHIFTED`, `FOURIER_<SEED>` (where <SEED> is the seed for sampling Fourier components).

NOISE_LEVELS is an ordered string of {2,3,4,5} (e.g., 2345, 25, 3, 345, etc), listing the included sensory noise magnitudes. 2 indicates high noise, 5 indicates low noise.
```


The command produces a file with path:

```
FILE = f"logs/CROSSVALID/{__file__.replace('_VIZ', '')}_{P}_{FOLD_HERE}_{REG_WEIGHT}_{GRID}.txt"
```

In the case `P=0`, the script is instead `SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_ZeroTrig.py`, with otherwise identical arguments.


This creates a file `logs/SIMULATED_REPLICATE/SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_<P>_<NOISE_LEVELS>_N<TRIALS>_<PRIOR>_<ENCODING>.txt`.

Example:

```
python3 SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py 
```
produces a dataset at
```
logs/SIMULATED_REPLICATE/SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt
```

You can use `CounterfactualModel_VIZ.py` to plot the model from which the data is simulated:
```
python CounterfactualModel_VIZ.py TODO
```
For a sample output, see TODO


## Fitting models

*See above for a concrete example.*

For P = 2, 4, 6, 8, you can fit the model at

```
python3 RunSynthetic_FreePrior_CosineLoss_OnSim.py <P> 0 10.0 180 <NAME_OF_DATASET>
```
where `logs/SIMULATED_REPLICATE/<NAME_OF_DATASET>` is the path of the file stored when simulating, and `<P>` is the exponent used for fitting. Note that this is can be distinct from the exponent used for simulating the dataset.

You can plot the model fit using
```
TODO
```
which produces

The NLL is stored at
```
TODO
```

The fitted parameters are stored at
```
TODO
```

There are specialized scripts for P=0 and P=1:

TODO


