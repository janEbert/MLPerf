#!/bin/bash
export TRAINING_INSTANCE_SIZE=128
export STAGE_DIR_PREFIX=/tmp/deepcam
#export STAGE_ONLY=1
export SEED="$(date +%s)"

./start_training_run.sh -s booster -N $((TRAINING_INSTANCE_SIZE/4)) -c ./configs/best_configs/config_DGXA100_128GPU_BS128_graph.sh -t 00:10:00
#configs/best_configs/config_DGXA100_128GPU_BS128_graph.sh  -t 01:15:00
