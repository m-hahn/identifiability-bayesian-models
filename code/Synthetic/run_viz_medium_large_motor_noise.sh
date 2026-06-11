#!/usr/bin/env bash
FITS=(
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_MediumLargeMotorNoise.py_180_2_2345_N10000_STEEPPERIODIC_STEEPPERIODIC.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_MediumLargeMotorNoise.py_180_2_2345_N10000_STEEPPERIODIC_UNIFORM.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_MediumLargeMotorNoise.py_180_2_2345_N10000_UNIFORM_STEEPPERIODIC.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_MediumLargeMotorNoise.py_180_2_2345_N10000_STEEPSHIFTED_STEEPPERIODIC.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_MediumLargeMotorNoise.py_180_8_2345_N10000_STEEPPERIODIC_STEEPPERIODIC.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_MediumLargeMotorNoise.py_180_8_2345_N10000_STEEPPERIODIC_UNIFORM.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_MediumLargeMotorNoise.py_180_8_2345_N10000_UNIFORM_STEEPPERIODIC.txt"
  "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_MediumLargeMotorNoise.py_180_8_2345_N10000_STEEPSHIFTED_STEEPPERIODIC.txt"
)

for FIT in "${FITS[@]}"; do
  if [[ "$FIT" == *_180_2_2345_* ]]; then
    P=2
  elif [[ "$FIT" == *_180_8_2345_* ]]; then
    P=8
  else
    echo "Unexpected FIT pattern: $FIT" >&2
    exit 1
  fi
  echo "VIZ P=$P FIT=$FIT"
  python3 RunSynthetic_FreePrior_CosineLoss_OnSim_VIZ_OnlyModel_OtherNoiseLevels.py "$P" 0 10.0 180 "$FIT"
done
