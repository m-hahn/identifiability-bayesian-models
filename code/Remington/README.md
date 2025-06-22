# Visualize simulated model

* [general](CounterfactualModel_BasedOnFit_Remington_VIZ.py)
* [for Figure 8](CounterfactualModel_BasedOnFit_Remington_VIZ_Figure8_OnlyEncPri.py)
* [general (with bias components)](CounterfactualModel_Remington_VIZ.py)
* [general (only encoding and prior)](CounterfactualModel_Remington_VIZ_Components.py)


# fitting on the original data
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

# Fitting on synthetic data
## fitting at p=1
* [fitting](RunSynthetic_DenseRemington_FreeEncoding_L1_OnSim_OtherNoiseLevels_VarySize.py)
* [(not used)](RunSynthetic_DenseRemington_FreeEncoding_L1_OnSim_OtherNoiseLevels_VarySize_Round2.py)
* [used in supplement to Figure 5](RunSynthetic_DenseRemington_FreeEncoding_L1_OnSim_OtherNoiseLevels_VarySize_VIZ.py)

## fitting
* [fitting](RunSynthetic_DenseRemington_FreeEncoding_OnSim_OtherNoiseLevels_VarySize.py)
* [(not used)](RunSynthetic_DenseRemington_FreeEncoding_OnSim_OtherNoiseLevels_VarySize_Round2.py)
* [used in supplement to Figure 5](RunSynthetic_DenseRemington_FreeEncoding_OnSim_OtherNoiseLevels_VarySize_VIZ.py)

## fitting at p=0
* [fitting (not used due to instability)](RunSynthetic_DenseRemington_FreeEncoding_Zero_OnSim_OtherNoiseLevels_VarySize.py)
* [fitting](RunSynthetic_DenseRemington_FreeEncoding_Zero_OnSim_OtherNoiseLevels_VarySize_Round2.py)
* [used in supplement to Figure 5](RunSynthetic_DenseRemington_FreeEncoding_Zero_OnSim_OtherNoiseLevels_VarySize_Round2_VIZ.py)


## fitting with Weber's law-based encoding
* [used for Figure 8, p>=2](RunSynthetic_DenseRemington_WeberEncoding_OnSim_OtherNoiseLevels_VarySize_Round2.py)
* [used for Figure 8](RunSynthetic_DenseRemington_WeberEncoding_Zero_OnSim_OtherNoiseLevels_VarySize_Round2.py), [visualization, used in Figure 8](RunSynthetic_DenseRemington_WeberEncoding_Zero_OnSim_OtherNoiseLevels_VarySize_Round2_VIZ_Figure8.py)

# Simulate
* [simulation on the basis of the original dataset (for Figure 8)](SimulateRemington_Lognormal_OtherNoiseLevels_Zero_DebugFurtherAug.py)
* [simulation for synthetic datasets](SimulateSynthetic2_DenseRemington_OtherNoiseLevels_VarySize.py)
* [batch script](SimulateSynthetic2_DenseRemington_OtherNoiseLevels_VarySize_ALL.py)

# utilities
* [synthetic prior and encoding](counterfactualComponents.py)


# Model fit statistics
* [used in Figure 8](evaluateCrossValidationResults_Remington_StopNoImpQ_OnlyFree.py)
* [used in supplement to Figure 5](evaluateCrossValidationResults_Synthetic_DenseRemington.py), [batch script](evaluateCrossValidationResults_Synthetic_DenseRemington_ALL.py)
* [used in Figure 8 (freely fitted encoding)](evaluateCrossValidationResults_Synthetic_Remington_Fig8.py)
* [used in Figure 8 (Weber's law-based encoding)](evaluateCrossValidationResults_Synthetic_Remington_Fig8_WeberEnc.py)
* [used in supplement to Figure 5](evaluateCrossValidationResults_Synthetic_Remington_VisualizeByNoiseCount_AndSize_ByP_Poster.py)

## Helpers
* [get observations](getObservations.py)
* [L1 estimator](l1IntervalEstimator.py)
* [load model](loadModel.py)
* [load data](loadRemington.py)
* [load simulated data](loadRemington_Simulated.py)
* [Lp estimator](lpEstimator.py)
* [L0](mapIntervalEstimator7.py)
* [L0](mapIntervalEstimator7_Debug.py)
* [L0](mapIntervalEstimator7_DebugFurther.py)
* [L0](mapIntervalEstimator7_DebugFurtherAug.py)
* [util](util.py)

# Batch scripts
* [TODO](runForFigure5.py)
* [TODO](runForFigure5_CheckProgress.py)
* [TODO](runForFigure5_L1.py)
* [TODO](runForFigure5_L1_Round2.py)
* [TODO](runForFigure5_Weber.py)
* [TODO](runForFigure5_Zero.py)
* [TODO](runForFigure6_Weber.py)

