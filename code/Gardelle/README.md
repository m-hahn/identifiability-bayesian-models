
# List of Files with Explanation
This file provides a catalogue of the relevant scripts in this directory. See [instructions for the behavioral data](../Synthetic/README.md#original-data).

## Fitting

This is from the [Hahn&Wei 2024 code base](https://gitlab.com/m-hahn/unifying-theory-biases/-/blob/main/code/Gardelle/RunGardelle_FreePrior_CosineLoss.py).

* [fitting on the original dataset](RunGardelle_FreePrior_CosineLoss.py)

## Tutorial: Running Fits on the Original de Gardelle et al. Data

Run all commands below from this directory.

### 1. Download the data

Download the original data file from the link provided by the original authors [on their website](https://sites.google.com/site/vincentdegardelle/publications?authuser=0).

The code expects the file at:

```text
data/GARDELLE/data.txt
```

Create the directory if needed, and place the downloaded `data.txt` there.

### 2. Fit the model on the full original dataset

The main fit used as the basis for the original-data analyses in the writeup is:

```bash
python3 RunGardelle_FreePrior_CosineLoss.py 8 0 10.0 180
```

This writes the fitted parameters to:

```text
logs/CROSSVALID/RunGardelle_FreePrior_CosineLoss.py_8_0_10.0_180.txt
```

and the loss to:

```text
losses/RunGardelle_FreePrior_CosineLoss.py_8_0_10.0_180.txt.txt
```

To visualize that fit, run:

```bash
python3 RunGardelle_FreePrior_CosineLoss_VIZFig7.py 8 0 10.0 180
```

This produces:

```text
figures/RunGardelle_FreePrior_CosineLoss_VIZFig7.py_8_0_10.0_180.pdf
```

This is the fit shown in the Figure 7 writeup file at `../../writeup/figure7a_byFI.tex`.

### 3. Reproduce the Figure 6 analyses on the original data

Figure 6 in the writeup (`../../writeup/figure6.tex`) uses 1,000-trial downsampled fits at each individual noise level, with `p=8` and seed `21`.

First fit the five downsampled datasets:

```bash
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize.py 8 0 10.0 180 1 1000 -21
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize.py 8 0 10.0 180 2 1000 -21
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize.py 8 0 10.0 180 3 1000 -21
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize.py 8 0 10.0 180 4 1000 -21
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize.py 8 0 10.0 180 5 1000 -21
```

Then create the panels used in Figure 6:

```bash
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncoding.py 8 0 10.0 180 1 1000 21
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncoding.py 8 0 10.0 180 2 1000 21
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncoding.py 8 0 10.0 180 3 1000 21
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncoding.py 8 0 10.0 180 4 1000 21
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncoding.py 8 0 10.0 180 5 1000 21

python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncodingNorm.py 8 0 10.0 180 1 1000 21
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncodingNorm.py 8 0 10.0 180 2 1000 21
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncodingNorm.py 8 0 10.0 180 3 1000 21
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncodingNorm.py 8 0 10.0 180 4 1000 21
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncodingNorm.py 8 0 10.0 180 5 1000 21

python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncodingTotalFI.py 8 0 10.0 180 1 1000 21
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncodingTotalFI.py 8 0 10.0 180 2 1000 21
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncodingTotalFI.py 8 0 10.0 180 3 1000 21
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncodingTotalFI.py 8 0 10.0 180 4 1000 21
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncodingTotalFI.py 8 0 10.0 180 5 1000 21
```

These write the PDFs referenced by `../../writeup/figure6.tex`.

### 4. Reproduce the Figure 7 loss-identification analyses

Figure 7 (`../../writeup/figure7a_byFI.tex`) combines the full-data fit above with a large collection of downsampled fits across exponents, seeds, trial counts, and noise-level subsets.

To generate the downsampled fitting runs used by that analysis, use the provided batch script:

```bash
python3 RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_All.py
```

This repeatedly launches fits for:

* exponents `0, 1, 2, 4, 6, 8`
* target sizes `1000`, `2000`, `10000`
* single-level and two-level subsets, plus some multi-level subsets
* seeds `21`, `22`, `23`, `24`

After enough of these runs have completed, collect and plot the comparison panels with:

```bash
python3 evaluateCrossValidationResults_Gardelle_180_Downsampled_TargetSize_Individual_ByFI.py
python3 colorLegendFigure7_ByFI.py
```

These write the subplot PDFs and legend used by `../../writeup/figure7a_byFI.tex`.

### 5. Notes

* Many scripts stop early, or assert, if their expected output file already exists. If you want to rerun a fit from scratch, move or delete the corresponding files in `logs/CROSSVALID/`, `losses/`, or `figures/`.
* The downsampling scripts use the integer noise-level labels `1, 2, 3, 4, 5` found in this codebase. For Figure 6, the writeup commands use the fixed seed `21`.
* Outputs for the original-data analyses are written locally in this directory, mainly under `logs/CROSSVALID/`, `losses/`, and `figures/`.

## Fitting on Downsampled Dataset (Material for Figures 6 and 7)

These are adapted to downsampled datasets.

* model fitting on donwsampled dattasets
  * [fit at p>=2](RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize.py), [batch script](RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_All.py), [plot model fit](RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_EncPri.py)
  * [fit at p=1, used for Figure 6](RunGardelle_FreePrior_L1Loss_Downsampled_TargetSize.py), [plot model fit](RunGardelle_FreePrior_L1Loss_Downsampled_TargetSize_VIZ_EncPri.py)
  * [fit at p=0, used for Figure 6](RunGardelle_FreePrior_ZeroTrig_Downsampled_TargetSize.py), [plot model fit](RunGardelle_FreePrior_ZeroTrig_Downsampled_TargetSize_VIZ_EncPri.py)
* visualizations for Figure 6: [unnormalized](RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncoding.py), [normalized](RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncodingNorm.py), [total FI](RunGardelle_FreePrior_CosineLoss_Downsampled_TargetSize_VIZ_OnlyEncodingTotalFI.py)
* visualizations for Figure 7:
  * [plot model fit, used for Figure 7](RunGardelle_FreePrior_CosineLoss_VIZFig7.py)
  * [color legend for Figure 7](colorLegendFigure7_ByFI.py)
  * [model fit statistics, used for Figure 7](evaluateCrossValidationResults_Gardelle_180_Downsampled_TargetSize_Individual_ByFI.py)

## Estimators
* [Lp estimator for circular space](cosineEstimator.py) (from Hahn&Wei 2024 code base)
* [L1](l1Estimator.py) (new code, as described in SI Appendix, Section S3.2)
* [L0](mapCircularEstimatorDebug.py) (based on Hahn&Wei 2024 code base, but improved implementation for circular space, as described in SI Appendix)

## Utilities
These are from the Hahn&Wei 2024 code base.
* [computations](computations.py)
* [get observations](getObservations.py)
* [load dataset](loadGardelle.py)
* [load model](loadModel.py)
* [util](util.py)
