#!/usr/bin/env bash
BASE_DIR="logs/SIMULATED_REPLICATE"

  find "$BASE_DIR" -type f -name '*_mean1_*' -name '*_sigma22_*' -print0 |
  python3 -c 'import sys, random
items = [x for x in sys.stdin.buffer.read().split(b"\0") if x]
random.shuffle(items)
sys.stdout.buffer.write(b"\0".join(items) + b"\0")
' |
  while IFS= read -r -d '' PATHNAME; do
    FILE="$(basename "$PATHNAME")"

  for P in 2 4 6 8; do
    echo "Running P=$P FILE=$FILE"

    python3 RunSynthetic_DenseRemington_FreeEncoding_OnSim_OtherNoiseLevels_VarySize_AcerbiLoss.py \
      "$P" 0 10.0 400 "$FILE"
  done
done
