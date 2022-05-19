#!/bin/bash

export INSTANCES=10
NODES_PER_INSTANCE=64
RESULTS_ROOT="/p/scratch/hai_mlperf/cosmoflow/results_weak/$INSTANCES/$NODES"
mkdir -p "$RESULTS_ROOT"
export SEED="$(date +%s)"

./start_training_run.sh --system booster \
    --nodes "$((INSTANCES * NODES_PER_INSTANCE))" \
    --time 02:00:00 \
    --config ../cosmoflow/configs/config_DGXA100_32x8x2_handrolled.sh
