#!/usr/bin/env bash
N_TRIALS="${N_TRIALS:-40000}"
DATA_PS=(2 8)
PRIORS=(STEEPPERIODIC STEEPPERIODIC UNIFORM STEEPSHIFTED)
ENCODINGS=(STEEPPERIODIC UNIFORM STEEPPERIODIC STEEPPERIODIC)
PS=(0 1 2 4 6 8)
TASKS=()

for DATA_P in "${DATA_PS[@]}"; do
  for IDX in "${!PRIORS[@]}"; do
    FIT="SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_MediumLargeMotorNoise.py_180_${DATA_P}_2345_N${N_TRIALS}_${PRIORS[$IDX]}_${ENCODINGS[$IDX]}.txt"
    for P in "${PS[@]}"; do
      TASKS+=("${P}|${FIT}")
    done
  done
done

mapfile -t SHUFFLED_TASKS < <(printf '%s\n' "${TASKS[@]}" | shuf)

for TASK in "${SHUFFLED_TASKS[@]}"; do
  IFS='|' read -r P FIT <<< "$TASK"
  echo "Training P=$P FIT=$FIT"
  case "$P" in
    0)
      python3 RunSynthetic_FreePrior_ZeroTrig_OnSim.py 0 0 10.0 180 "$FIT"
      ;;
    1)
      python3 RunSynthetic_FreePrior_L1Loss_OnSim.py 1 0 10.0 180 "$FIT"
      ;;
    *)
      python3 RunSynthetic_FreePrior_CosineLoss_OnSim.py "$P" 0 10.0 180 "$FIT"
      ;;
  esac
done
