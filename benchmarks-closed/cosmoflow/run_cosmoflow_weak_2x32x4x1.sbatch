#!/bin/bash

export INSTANCES=2
NODES=32
RESULTS_ROOT="/p/scratch/jb_benchmark/cosmoflow/results_weak/$INSTANCES/$NODES"
mkdir -p "$RESULTS_ROOT"

# Default is 50
export NUMEPOCHS=50

./start_training_run.sh --system booster --nodes "$((INSTANCES * NODES))" \
    --time 02:00:00 --config cosmoflow/configs/config_DGXA100_common.sh
