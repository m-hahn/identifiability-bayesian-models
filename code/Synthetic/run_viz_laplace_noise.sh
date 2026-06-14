#!/usr/bin/env bash
FITS=(
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py_180_2_2345_N10000_STEEPPERIODIC_STEEPPERIODIC.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py_180_2_2345_N10000_STEEPPERIODIC_UNIFORM.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py_180_2_2345_N10000_UNIFORM_STEEPPERIODIC.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py_180_2_2345_N10000_STEEPSHIFTED_STEEPPERIODIC.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py_180_8_2345_N10000_STEEPPERIODIC_STEEPPERIODIC.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py_180_8_2345_N10000_STEEPPERIODIC_UNIFORM.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py_180_8_2345_N10000_UNIFORM_STEEPPERIODIC.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py_180_8_2345_N10000_STEEPSHIFTED_STEEPPERIODIC.txt"
)

for FIT in "${FITS[@]}"; do
  for P in 0 1 2 4 6 8; do
    echo "VIZ P=$P FIT=$FIT"
    case "$P" in
      0)
        python3 RunSynthetic_FreePrior_ZeroTrig_OnSim_LaplaceNoise_VIZ_OnlyModel.py 0 0 10.0 180 "$FIT"
        ;;
      1)
        python3 RunSynthetic_FreePrior_L1Loss_OnSim_LaplaceNoise_VIZ_OnlyModel.py 1 0 10.0 180 "$FIT"
        ;;
      *)
        python3 RunSynthetic_FreePrior_CosineLoss_OnSim_LaplaceNoise_VIZ_OnlyModel.py "$P" 0 10.0 180 "$FIT"
        ;;
    esac
  done
done
