# List of Files with Explanation

## Plotting simulated models
* [show a simulated model](CounterfactualModel_VIZ.py)
* [show a simulated model (without attraction and repulstion components)](CounterfactualModel_VIZNoAttRep.py)
* [show simulated model, with prior transformed based on loss function](CounterfactualModel_VIZ_ByNoiseMagnGauge.py)
* [show a simulated model (only prior and encoding)](CounterfactualModel_VIZ_Components.py)
* [show a simulated model (only prior and encoding)](CounterfactualModel_VIZ_Components_Fig2.py)
* [show a simulated model (only prior and encoding)](CounterfactualModel_VIZ_Components_Fig5.py)
* [show simulated model (with stimulus space)](CounterfactualModel_VIZ_WithStimNoise.py)



## Fitting Model on siulated data

* [basic script for fitting model (Losses with p >= 2)](RunSynthetic_FreePrior_CosineLoss_OnSim.py)

### Utilities for visualization
* [visualizing fit](RunSynthetic_FreePrior_CosineLoss_OnSim_VIZ.py)
* [visualizing fit (without attraction and repulsion components)](RunSynthetic_FreePrior_CosineLoss_OnSim_VIZNoAttRep.py)
* [visualizing fit (only prior and encoding)](RunSynthetic_FreePrior_CosineLoss_OnSim_VIZ_OnlyModel.py)
* [visualizing fit (only prior and encoding)](RunSynthetic_FreePrior_CosineLoss_OnSim_VIZ_OnlyModel_OtherNoiseLevels.py)
* [visualizing fit (only prior and encoding, used for Figure 2)](RunSynthetic_FreePrior_CosineLoss_OnSim_VIZ_OnlyModel_OtherNoiseLevels_Fig2.py)
* [visualizing fit (only prior and encoding, used for Figure 3)](RunSynthetic_FreePrior_CosineLoss_OnSim_VIZ_OnlyModel_OtherNoiseLevels_Figure3.py), [visualization used in Figure 3 (p=1)](RunSynthetic_FreePrior_L1Loss_OnSim_VIZ_OnlyModel_OtherNoiseLevels.py), [used in Figure 3, (p=0)](RunSynthetic_FreePrior_ZeroTrig_OnSim_VIZ_OnlyModel_OtherNoiseLevels_Figure3.py)
* [visualizing fit (only prior and encoding, including ground truth prior)](RunSynthetic_FreePrior_CosineLoss_OnSim_VIZ_OnlyModel_OtherNoiseLevels_WithGroundTruthPrior.py)
* [applying Theorem 1](recover_encoding.py)

### With L1 loss
* [fitting model](RunSynthetic_FreePrior_L1Loss_OnSim.py)
* [visualization](RunSynthetic_FreePrior_L1Loss_OnSim_VIZ.py), [visualization without attraction/repulsion](RunSynthetic_FreePrior_L1Loss_OnSim_VIZNoAttRep.py)

### With L0 loss
* [fitting model](RunSynthetic_FreePrior_ZeroTrig_OnSim.py)
* [showing fit](RunSynthetic_FreePrior_ZeroTrig_OnSim_VIZ.py)


### With separate encoding per noise level

* [batch script for fitting model](runForFigure5_AdditiveEncoding.py)
* [separate encodings per noise level](CounterfactualModel_AdditiveEncodings_VIZ.py)
* [separate encodings per noise level](CounterfactualModel_SeparateEncodings_VIZ.py)
* [basic script for fitting model with separate encoding per noise level (losses with p >= 2)](RunSynthetic_FreePrior_CosineLoss_OnSim_SeparateEncoding.py), [loss p=1](RunSynthetic_FreePrior_L1Loss_OnSim_SeparateEncoding.py), [loss p=0](RunSynthetic_FreePrior_ZeroTrig_OnSim_SeparateEncoding.py)
* [visualizing fit with separate encoding per noise level](RunSynthetic_FreePrior_CosineLoss_OnSim_SeparateEncoding_VIZ.py), [visualizing p=1](RunSynthetic_FreePrior_L1Loss_OnSim_SeparateEncoding_VIZ.py), [visualizing p=0](RunSynthetic_FreePrior_ZeroTrig_OnSim_SeparateEncoding_VIZ.py)
* [batch script for fitting (p>=2)](runForFigure5_SeparateEncoding.py), [batch script for fitting (p=1)](runForFigure5_SeparateEncoding_L1.py),  [batch script for fitting (p=0)](runForFigure5_SeparateEncoding_Zero.py)
* [simulate](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_AdditiveEncodings.py)
* [batch script](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_AdditiveEncodings_ALL.py)
* [simulate](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_SeparateEncodings.py)
* [simulate (p=0)](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_ZeroTrig_AdditiveEncodings.py)
* [simulate (p=0)](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_ZeroTrig_SeparateEncodings.py)
* [separate encodings](evaluateCrossValidationResults_Synthetic_Gardelle_NonF_SeparateEncoding.py)
* [batch script](evaluateCrossValidationResults_Synthetic_Gardelle_NonF_SeparateEncoding_ALL.py)
* [collecting model fit statistics for separate encodings (short file name version due to Unix file name length limit)](evaluateCross_BRIEF_SeparateEncoding.py)


### Including Stimulus Noise
* [fitting (including stimulus noise, losses with p >= 2)](RunSynthetic_FreePrior_CosineLoss_OnSim_WithStimNoise.py), [fitting at p=1](RunSynthetic_FreePrior_L1Loss_OnSim_WithStimNoise.py), [fitting at p=0](RunSynthetic_FreePrior_ZeroTrig_OnSim_WithStimNoise.py), [batch script](runForFigure5_StimNoise.py)
* [showing fit](RunSynthetic_FreePrior_CosineLoss_OnSim_WithStimNoise_VIZ.py)



## Simulate Datasets
* [simulating dataset](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py)
* [batch script](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_ALL.py)
* [batch script (p=1)](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_L1_ALL.py)
* [at p=0](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_ZeroTrig.py)
* [batch script](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_ZeroTrig_ALL.py)
* [simulate with stimulus noise](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_WithStimNoise.py)

### 2AFC
* [simulate](Simulate_2AFC_Synthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_WithKL.py)
* [simulate (noise-less reference)](Simulate_2AFC_Synthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_WithKL_CleanRef.py)
* [fitting](Run_2AFC_Synthetic_FreePrior_CosineLoss_OnSim.py)
* [fitting (noise-less reference)](Run_2AFC_Synthetic_FreePrior_CosineLoss_OnSim_CleanRef.py)
* [batch script](Run_2AFC_Synthetic_FreePrior_CosineLoss_OnSim_CleanRef_RUNALL.py)
* [visualization](Run_2AFC_Synthetic_FreePrior_CosineLoss_OnSim_CleanRef_VIZ.py)
* [batch script](Run_2AFC_Synthetic_FreePrior_CosineLoss_OnSim_RUNALL.py)
* [visualization](Run_2AFC_Synthetic_FreePrior_CosineLoss_OnSim_VIZ.py)
* [fitting at p=1](Run_2AFC_Synthetic_FreePrior_L1Loss_OnSim.py)
* [fitting at p=0](Run_2AFC_Synthetic_FreePrior_ZeroTrig_OnSim.py)

## Collecting Model Fit statistics
* [(used for Figure 4)](evaluateCrossValidationResults_Synthetic_Gardelle.py)
* [for 2AFC](evaluateCrossValidationResults_Synthetic_Gardelle_2AFC.py)
* [for 2AFC (with noise-less reference)](evaluateCrossValidationResults_Synthetic_Gardelle_2AFC_CleanRef.py)
* [batch script](evaluateCrossValidationResults_Synthetic_Gardelle_ALL.py)
* [visualization of fit](evaluateCrossValidationResults_Synthetic_Gardelle_NonF.py)
* [batch script](evaluateCrossValidationResults_Synthetic_Gardelle_NonF_ALL.py)
* [(used in Figure 3)](evaluateCrossValidationResults_Synthetic_Gardelle_NonF_Figure3.py)
* [stimulus noise](evaluateCrossValidationResults_Synthetic_Gardelle_NonF_StimNoise.py)
* [possibly used for Figure 5](evaluateCrossValidationResults_Synthetic_Gardelle_VisualizeByNoiseCount_AndSize_ByP_ConfusMat.py)
* [possibly used for Figure 5](evaluateCrossValidationResults_Synthetic_Gardelle_VisualizeByNoiseCount_AndSize_ByP_JustCollStat.py)
* [used for supplement to Figure 5](evaluateCrossValidationResults_Synthetic_Gardelle_VisualizeByNoiseCount_AndSize_ByP_Poster_Exculde1.py)
* [used for Figure 5](evaluateCrossValidationResults_Synthetic_Gardelle_VisualizeByNoiseCount_AndSize_ByP_Poster_Exculde1_Figure5.py)

## Estimators
The L1 estimator is newly added (see SI Appendix for description). The other estimates are from the codebase of Hahn&Wei 2024.
* [L1 estimator](l1Estimator.py)
* [L0 estimator](mapCircularEstimator10.py), [improved implementation eliminating fitting artifacts at boundary](mapCircularEstimatorDebug.py)
* [Lp estimator at p>=2 (circular spaces)](cosineEstimator.py), [variant (with clamped Newton updates, sometimes prevents numerical instability)](cosineEstimator6.py)

## Batch script for Figure 5 (and associated figures in SI Appendix)
* [basic](runForFigure5.py)
* [p=1](runForFigure5_L1.py)
* [p=2](runForFigure5_OnlyL2.py)
* [p=0](runForFigure5_Zero.py)

## Utilities
These are helpers from the codebase of Hahn&Wei 2024.
* [computations](computations.py)
* [load de Gardelle et al data](loadGardelle.py)
* [load model fit](loadModel.py)
* [retrieve observations](getObservations.py)
* [util](util.py)
* [code for the synthetic priors and encodings](counterfactualComponents.py)




## Note
The data from de Gardelle et al 2010 was provided by the original authors at https://sites.google.com/site/vincentdegardelle/publications
