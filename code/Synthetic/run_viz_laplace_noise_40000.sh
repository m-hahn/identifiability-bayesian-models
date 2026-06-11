#!/usr/bin/env bash
N_TRIALS="${N_TRIALS:-40000}"
PYTHON_BIN="${PYTHON_BIN:-/Users/mhahn/anaconda3/bin/python}"
FITS=(
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py_180_2_2345_N${N_TRIALS}_STEEPPERIODIC_STEEPPERIODIC.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py_180_2_2345_N${N_TRIALS}_STEEPPERIODIC_UNIFORM.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py_180_2_2345_N${N_TRIALS}_UNIFORM_STEEPPERIODIC.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py_180_2_2345_N${N_TRIALS}_STEEPSHIFTED_STEEPPERIODIC.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py_180_8_2345_N${N_TRIALS}_STEEPPERIODIC_STEEPPERIODIC.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py_180_8_2345_N${N_TRIALS}_STEEPPERIODIC_UNIFORM.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py_180_8_2345_N${N_TRIALS}_UNIFORM_STEEPPERIODIC.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py_180_8_2345_N${N_TRIALS}_STEEPSHIFTED_STEEPPERIODIC.txt"
)

for FIT in "${FITS[@]}"; do
  DATA_P="$(echo "$FIT" | sed -E 's/.*_180_([0-9]+)_2345_.*/\1/')"
  echo "VIZ laplace-on-laplace P=$DATA_P FIT=$FIT"
  "$PYTHON_BIN" RunSynthetic_FreePrior_CosineLoss_OnSim_LaplaceNoise_VIZ_OnlyModel.py "$DATA_P" 0 10.0 180 "$FIT"
done
