#!/bin/bash
export TRAINING_INSTANCE_SIZE=4
export STAGE_DIR_PREFIX=/tmp/deepcam
export SEED="$(date +%s)"

./start_training_run.sh -s booster -N $((TRAINING_INSTANCE_SIZE/4)) -c ./configs/test_configs/config_DGXA100_8GPU_BS128_dummy_graph.sh  -t 00:15:00
