# List of Files with Explanation

## Basic Modeling Framewotk

### Plotting simulated models
The main script is:
* [show a simulated model](CounterfactualModel_VIZ.py) (see [more here](../README.md#Instructions))
  
The following variants differ in what is plotted or the visual formatting, but generally use the same command line arguments. You can use tools such as `vimdiff` to see how these files differ in implementation.
* [show a simulated model (without attraction and repulsion components, e.g. for Figure 4)](CounterfactualModel_VIZNoAttRep.py)
* [show simulated model, with prior transformed based on loss function, for illustrating confoundedness of prior and loss function in SI Appendix, Figure S12](CounterfactualModel_VIZ_ByNoiseMagnGauge.py)
* [show a simulated model (only prior and encoding)](CounterfactualModel_VIZ_Components.py), [version for Figure 2](CounterfactualModel_VIZ_Components_Fig2.py), [version for Figure 5](CounterfactualModel_VIZ_Components_Fig5.py)


### Simulate Datasets
* [simulating dataset at p>0](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py),  [at p=0](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_ZeroTrig.py)   (see [more here](../README.md#Instructions))
* [batch script (p>1)](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_ALL.py), [batch script (p=1)](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_L1_ALL.py), [batch script (p=0)](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_ZeroTrig_ALL.py)


### Fitting Model on simulated data
The main script is:
* [basic script for fitting model (Losses with p >= 2)](RunSynthetic_FreePrior_CosineLoss_OnSim.py) (see [more here](../README.md#Instructions)), [version at p=1](RunSynthetic_FreePrior_L1Loss_OnSim.py), [version at p=0](RunSynthetic_FreePrior_ZeroTrig_OnSim.py)

### Utilities for visualizing fitted models
The main script is:
* [visualizing fit](RunSynthetic_FreePrior_CosineLoss_OnSim_VIZ.py)  (see [../README.md#Instructions](for more)), [version at p=1](RunSynthetic_FreePrior_L1Loss_OnSim_VIZ.py), [version at p=0](RunSynthetic_FreePrior_ZeroTrig_OnSim_VIZ.py)
The following variants differ in what is plotted or the visual formatting. You can use tools such as `vimdiff` to see how these files differ in implementation.
* [visualizing fit (without attraction and repulsion components)](RunSynthetic_FreePrior_CosineLoss_OnSim_VIZNoAttRep.py), [version at p=1](RunSynthetic_FreePrior_L1Loss_OnSim_VIZNoAttRep.py)
* [visualizing fit (only prior and encoding)](RunSynthetic_FreePrior_CosineLoss_OnSim_VIZ_OnlyModel.py)
* [visualizing fit (only prior and encoding)](RunSynthetic_FreePrior_CosineLoss_OnSim_VIZ_OnlyModel_OtherNoiseLevels.py)
* [visualizing fit (only prior and encoding, used for Figure 2)](RunSynthetic_FreePrior_CosineLoss_OnSim_VIZ_OnlyModel_OtherNoiseLevels_Fig2.py)
* [visualizing fit (only prior and encoding, used for Figure 3)](RunSynthetic_FreePrior_CosineLoss_OnSim_VIZ_OnlyModel_OtherNoiseLevels_Figure3.py), [visualization used in Figure 3 (p=1)](RunSynthetic_FreePrior_L1Loss_OnSim_VIZ_OnlyModel_OtherNoiseLevels.py), [used in Figure 3, (p=0)](RunSynthetic_FreePrior_ZeroTrig_OnSim_VIZ_OnlyModel_OtherNoiseLevels_Figure3.py)
* [visualizing fit (only prior and encoding, including ground truth prior)](RunSynthetic_FreePrior_CosineLoss_OnSim_VIZ_OnlyModel_OtherNoiseLevels_WithGroundTruthPrior.py)


### Collecting Model Fit statistics
TODO add epxlainer
* [collecting NLL](evaluateCrossValidationResults_Synthetic_Gardelle.py), 
* [batch script](evaluateCrossValidationResults_Synthetic_Gardelle_ALL.py)
* [visualization of fit](evaluateCrossValidationResults_Synthetic_Gardelle_NonF.py), [(used in Figure 3)](evaluateCrossValidationResults_Synthetic_Gardelle_NonF_Figure3.py)
* [batch script](evaluateCrossValidationResults_Synthetic_Gardelle_NonF_ALL.py) 
* [stimulus noise](evaluateCrossValidationResults_Synthetic_Gardelle_NonF_StimNoise.py)
* [possibly used for Figure 5](evaluateCrossValidationResults_Synthetic_Gardelle_VisualizeByNoiseCount_AndSize_ByP_ConfusMat.py)
* [possibly used for Figure 5](evaluateCrossValidationResults_Synthetic_Gardelle_VisualizeByNoiseCount_AndSize_ByP_JustCollStat.py)
* [used for supplement to Figure 5](evaluateCrossValidationResults_Synthetic_Gardelle_VisualizeByNoiseCount_AndSize_ByP_Poster_Exculde1.py)
* [used for Figure 5](evaluateCrossValidationResults_Synthetic_Gardelle_VisualizeByNoiseCount_AndSize_ByP_Poster_Exculde1_Figure5.py)
* [used for Figure 4](evaluateCrossValidationResults_Synthetic_Gardelle_Figure4.py)

## Other Situations (SI Appendix, Section S5)



### 2AFC
The following scripts are relevant to SI Appendix, Section S5.1. We distinguish two versions: with the reference noised (default) or not (noise-less reference).
* simulate
  - [simulate](Simulate_2AFC_Synthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_WithKL.py)
  - [simulate (noise-less reference)](Simulate_2AFC_Synthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_WithKL_CleanRef.py)
* fitting
  * [fitting (reference subject to noise)](Run_2AFC_Synthetic_FreePrior_CosineLoss_OnSim.py), [batch script](Run_2AFC_Synthetic_FreePrior_CosineLoss_OnSim_CleanRef_RUNALL.py), [visualization](Run_2AFC_Synthetic_FreePrior_CosineLoss_OnSim_CleanRef_VIZ.py), [fitting at p=1](Run_2AFC_Synthetic_FreePrior_L1Loss_OnSim.py), [fitting at p=0](Run_2AFC_Synthetic_FreePrior_ZeroTrig_OnSim.py)
  * [fitting (noise-less reference)](Run_2AFC_Synthetic_FreePrior_CosineLoss_OnSim_CleanRef.py), [batch script](Run_2AFC_Synthetic_FreePrior_CosineLoss_OnSim_RUNALL.py), [visualization](Run_2AFC_Synthetic_FreePrior_CosineLoss_OnSim_VIZ.py)
* collecting NLL
  * [reference subject to noise](evaluateCrossValidationResults_Synthetic_Gardelle_2AFC.py)
  * [with noise-less reference](evaluateCrossValidationResults_Synthetic_Gardelle_2AFC_CleanRef.py)



### Including Stimulus Noise
The following scripts are relevant to SI Appendix, Section S5.2.
* [fitting (including stimulus noise, losses with p >= 2)](RunSynthetic_FreePrior_CosineLoss_OnSim_WithStimNoise.py), [fitting at p=1](RunSynthetic_FreePrior_L1Loss_OnSim_WithStimNoise.py), [fitting at p=0](RunSynthetic_FreePrior_ZeroTrig_OnSim_WithStimNoise.py), [batch script](runForFigure5_StimNoise.py)
* [showing fit](RunSynthetic_FreePrior_CosineLoss_OnSim_WithStimNoise_VIZ.py)
* [simulate dataset](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_WithStimNoise.py)
* [show simulated model](CounterfactualModel_VIZ_WithStimNoise.py)

### With separate encoding per noise level
The following scripts are relevant to SI Appendix, Section S5.3.
* simulate data
  * [simulate (additively related encodings)](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_AdditiveEncodings.py),  [simulate (p=0)](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_ZeroTrig_AdditiveEncodings.py), [batch script](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_AdditiveEncodings_ALL.py)
  * [simulate data (separate random encodings)](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_SeparateEncodings.py), [simulate (p=0)](SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_ZeroTrig_SeparateEncodings.py)
* plotting model used for simulations
  * [plotting model used for simulations: separate encodings per noise level (additively related encodings)](CounterfactualModel_AdditiveEncodings_VIZ.py)
  * [plotting model used for simulations: separate encodings per noise level (separate random encodings)](CounterfactualModel_SeparateEncodings_VIZ.py)
- fit model with separate encoding per noise level
  - [basic script for fitting model with separate encoding per noise level (losses p >= 2)](RunSynthetic_FreePrior_CosineLoss_OnSim_SeparateEncoding.py), [loss p=1](RunSynthetic_FreePrior_L1Loss_OnSim_SeparateEncoding.py), [at p=0](RunSynthetic_FreePrior_ZeroTrig_OnSim_SeparateEncoding.py)
  - [batch script for fitting (p>=2)](runForFigure5_SeparateEncoding.py), [at p=1](runForFigure5_SeparateEncoding_L1.py),  [at p=0](runForFigure5_SeparateEncoding_Zero.py)
- [visualizing fit](RunSynthetic_FreePrior_CosineLoss_OnSim_SeparateEncoding_VIZ.py), [at p=1](RunSynthetic_FreePrior_L1Loss_OnSim_SeparateEncoding_VIZ.py), [at p=0](RunSynthetic_FreePrior_ZeroTrig_OnSim_SeparateEncoding_VIZ.py)
- [collect NLL](evaluateCrossValidationResults_Synthetic_Gardelle_NonF_SeparateEncoding.py), [batch script](evaluateCrossValidationResults_Synthetic_Gardelle_NonF_SeparateEncoding_ALL.py), [collecting model fit statistics for separate encodings (short file name version due to Unix file name length limit)](evaluateCross_BRIEF_SeparateEncoding.py)


## Utilities
### Estimators
This is generally based on the codebase of Hahn&Wei 2024. The L1 estimator is newly added (see SI Appendix S3.2 for description). 
* [L1 estimator](l1Estimator.py)
* [L0 estimator](mapCircularEstimator10.py), [improved implementation eliminating fitting artifacts at boundary](mapCircularEstimatorDebug.py) (see SI Appendix S3.1)
* [Lp estimator at p>=2 (circular spaces)](cosineEstimator.py), [variant (with clamped Newton updates, sometimes prevents numerical instability)](cosineEstimator6.py).

### Batch scripts for Figures 4, 5 (and associated figures in SI Appendix)
* Figure 4
  * [run training for Figure 4](run_cross_fitting_fig4.sh), [run training for Figure 4](run_ground_truth_fitting_fig4.sh)
  * [create synthetic datasets for Figure 4](run_data_sampling_fig4.sh)
* Figure 5
  * [overall batch script](run_training_fig5.sh), [p>=2](runForFigure5.py), [p=1](runForFigure5_L1.py), [p=2](runForFigure5_OnlyL2.py), [p=0](runForFigure5_Zero.py)
  * [create synthetic datasets for Figure 5](run_data_sampling_fig5.sh)
  * [collect model fit](run_evaluate_fig5.sh)

### Utilities
These are helpers from the codebase of Hahn&Wei 2024. These are imported by the other scripts, and do not need to be called by the user.
* [auxiliary computations](computations.py)
* [load de Gardelle et al data](loadGardelle.py)
* [load model fit](loadModel.py)
* [retrieve observations](getObservations.py)
* [util](util.py)

The following utility stores the different counterfactual priors and encodings:
* [code for the synthetic priors and encodings](counterfactualComponents.py)  (see [more here](../README.md#Instructions))


### Original Data
The data from de Gardelle et al 2010 was provided by the original authors at https://sites.google.com/site/vincentdegardelle/publications
The file `data.txt` is expected at `data/GARDELLE/data.txt`.

## Miscellaneous 
* [Comparing methods for FI computation (SI Appendix, Figure S2 top)](CounterfactualModel_VIZ_CheckFI.py)
* [applying Theorem 1 (used for illustrating Theorem 1, SI Appendix, Figure S3)](recover_encoding.py)



