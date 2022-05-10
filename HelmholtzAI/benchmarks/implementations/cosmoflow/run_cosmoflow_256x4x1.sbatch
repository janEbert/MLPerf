#!/bin/bash

NODES=256
RESULTS_ROOT="/p/scratch/hai_mlperf/cosmoflow/results/$NODES"
mkdir -p "$RESULTS_ROOT"

./start_training_run.sh --system booster --nodes "$NODES" \
    --time 01:00:00 \
    --config cosmoflow/configs/config_DGXA100_128x8x1.sh
