#!/bin/bash

NODES=128
RESULTS_ROOT="/p/scratch/hai_mlperf/cosmoflow/results/$NODES"
mkdir -p "$RESULTS_ROOT"
export SEED="$(date +%s)"

./start_training_run.sh --system booster --nodes "$NODES" \
    --time 01:00:00 \
    --config ../cosmoflow/configs/config_DGXA100_64x8x1_reference.sh
