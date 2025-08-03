
# List of Files with Explanation
This file provides a catalogue of the relevant scripts in this directory. See [instructions for the behavioral data](../Synthetic/README.md#original-data).

## Fitting

This is from the Hahn&Wei 2024 code base.

* [fitting on the original dataset](RunGardelle_FreePrior_CosineLoss.py)

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
