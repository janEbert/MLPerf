#!/bin/bash

NODES=64
RESULTS_ROOT="/p/scratch/jb_benchmark/cosmoflow/results/$NODES"
mkdir -p "$RESULTS_ROOT"

# Default is 50
export NUMEPOCHS=50

./start_training_run.sh --system booster --nodes "$NODES" \
    --time 01:00:00 --config cosmoflow/configs/config_DGXA100_common.sh
