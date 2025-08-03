# List of Files with Explanation

This file provides a catalogue of the relevant scripts in this directory, in analogy to those for [circular stimulus spaces](../Synthetic/README.md) and the [dataset from de Gardelle et al 2011](../Gardelle/README.md).

Various scripts expect access to the dataset from Remington et al 2018, see [instructions](#original-data).


## Basic Modeling Framework
We provide code for two sets of simulations: simulated variants of behavioral dataset (Figure 8), and generic simulations (SI Appendix, Section S4.3).

### A: Visualize simulated model

* simulated variants of behavioral dataset (Figure 8): [general](CounterfactualModel_BasedOnFit_Remington_VIZ.py), [for Figure 8](CounterfactualModel_BasedOnFit_Remington_VIZ_Figure8_OnlyEncPri.py)
* models for SI Appendix, Section S4.3: [general (with bias components)](CounterfactualModel_Remington_VIZ.py), [general (only encoding and prior)](CounterfactualModel_Remington_VIZ_Components.py)


### B: Simulate
* [simulated variants of behavioral dataset (Figure 8)](SimulateRemington_Lognormal_OtherNoiseLevels_Zero_DebugFurtherAug.py)
* [generic datasets](SimulateSynthetic2_DenseRemington_OtherNoiseLevels_VarySize.py), [batch script](SimulateSynthetic2_DenseRemington_OtherNoiseLevels_VarySize_ALL.py)

### C: Fitting on synthetic data
* fitting at p=1: [fitting](RunSynthetic_DenseRemington_FreeEncoding_L1_OnSim_OtherNoiseLevels_VarySize.py), [with rounding (SI Appendix, Section S3.1)](RunSynthetic_DenseRemington_FreeEncoding_L1_OnSim_OtherNoiseLevels_VarySize_Round2.py)
* fitting at p>=2: [fitting](RunSynthetic_DenseRemington_FreeEncoding_OnSim_OtherNoiseLevels_VarySize.py), [with rounding (SI Appendix, Section S3.1)](RunSynthetic_DenseRemington_FreeEncoding_OnSim_OtherNoiseLevels_VarySize_Round2.py)
* fitting at p=0: [fitting](RunSynthetic_DenseRemington_FreeEncoding_Zero_OnSim_OtherNoiseLevels_VarySize.py), [with rounding (SI Appendix, Section S3.1)](RunSynthetic_DenseRemington_FreeEncoding_Zero_OnSim_OtherNoiseLevels_VarySize_Round2.py)

### D: Visualizing Model Fit
* [p>=2](RunSynthetic_DenseRemington_FreeEncoding_OnSim_OtherNoiseLevels_VarySize_VIZ.py)
* [p=1](RunSynthetic_DenseRemington_FreeEncoding_L1_OnSim_OtherNoiseLevels_VarySize_VIZ.py)
* [p=0](RunSynthetic_DenseRemington_FreeEncoding_Zero_OnSim_OtherNoiseLevels_VarySize_Round2_VIZ.py)
  
### E: Model fit statistics
These scripts collects NLL statistics for fits on a given dataset across loss functions. Variants differ in which versions of the fitting procedure they apply to, and the formatting of resulting plots.
* [used in Figure 8](evaluateCrossValidationResults_Remington_StopNoImpQ_OnlyFree.py)
* [used in SI Appendix, Section S4.3](evaluateCrossValidationResults_Synthetic_DenseRemington.py), [batch script](evaluateCrossValidationResults_Synthetic_DenseRemington_ALL.py)
* [used in Figure 8 (freely fitted encoding)](evaluateCrossValidationResults_Synthetic_Remington_Fig8.py)
* [used in Figure 8 (Weber's law-based encoding)](evaluateCrossValidationResults_Synthetic_Remington_Fig8_WeberEnc.py)
* jointly plotting results for a combination of prior and encoding, across trial counts, noise levels, and loss functions: [used in SI Appendix, Section S4.3](evaluateCrossValidationResults_Synthetic_Remington_VisualizeByNoiseCount_AndSize_ByP_Poster.py)


## fitting with Weber's law-based encoding
* [used for Figure 8, p>=2](RunSynthetic_DenseRemington_WeberEncoding_OnSim_OtherNoiseLevels_VarySize_Round2.py)
* [used for Figure 8](RunSynthetic_DenseRemington_WeberEncoding_Zero_OnSim_OtherNoiseLevels_VarySize_Round2.py), [visualization, used in Figure 8](RunSynthetic_DenseRemington_WeberEncoding_Zero_OnSim_OtherNoiseLevels_VarySize_Round2_VIZ_Figure8.py)

# Fitting on Original Data
## fitting on the original data
* [visualize original data](RunRemington_Free_Zero_VIZ_ForFig8_Human.py)

## Fitting with Weber's law-based encoding
* [fitting at p>=2, used in Figure 8](RunRemington_Free.py)
* [fitting at p=0, used in Figure 8](RunRemington_Free_Zero.py), [visualization in Figure 8](RunRemington_Free_Zero_VIZ_ForFig8.py)

## Fitting with log-normal prior (alternative, reported in SI Appendix)
* [fitting at p=0](RunRemington_Lognormal_Zero.py), [visualization](RunRemington_Lognormal_Zero_VIZ.py)
* [fitting at p>=2](RunRemington_Lognormal.py)
* [visualize original data](RunRemington_Lognormal_VIZ_OnlyHuman_MainPaper_ErrBar.py)
* [visualize simulated data](RunRemington_Lognormal_VIZ_OnlySimulated_MainPaper_ErrBar.py)


## Fitting with free encoding (used in SI Appendix)

* [Lp](RunRemington_Free_FreeEncoding.py), [alternative with discretization (equivalent results at such p's)](RunRemington_Free_FreeEncoding_Round2.py)
* [L1 (with discretization)](RunRemington_Free_FreeEncoding_L1_Round2.py) 
* [L0](RunRemington_Free_FreeEncoding_Zero_Round2.py)

## Miscellanuous

### Estimators
This is generally based on the codebase of Hahn&Wei 2024. The L1 estimator is newly added (see SI Appendix S3.2 for description).

* [L1 estimator](l1IntervalEstimator.py)
* [Lp estimator](lpEstimator.py)
* [L0](mapIntervalEstimator7.py)
* [L0](mapIntervalEstimator7_Debug.py)
* [L0](mapIntervalEstimator7_DebugFurther.py)
* [L0](mapIntervalEstimator7_DebugFurtherAug.py)

### Batch scripts for SI Appendix, Section S4.3
* These scripts perform fitting for SI Appendix, Section S4.3: [p>=2](runForFigure5.py), [p>=2](runForFigure5_CheckProgress.py), [p=1](runForFigure5_L1.py), [p=1](runForFigure5_L1_Round2.py), [p=0](runForFigure5_Zero.py)


### Utilities
These are helpers, largely from the codebase of Hahn&Wei 2024. These are imported by the other scripts, and do not need to be called by the user.

* [get observations](getObservations.py)
* [load model](loadModel.py)
* [load data](loadRemington.py)
* [load simulated data](loadRemington_Simulated.py)
* [util](util.py)

The following utility stores the different counterfactual priors and encodings:
* [synthetic prior and encoding](counterfactualComponents.py)


## Original Data

The files are expected at `data/REMINGTON/`. In particular, the files `data/REMINGTON/Datafiles/*.mat` are read by the code.
