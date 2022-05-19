#!/bin/bash
export TRAINING_INSTANCE_SIZE=2048
export STAGE_DIR_PREFIX=/tmp/deepcam
export STAGE_ONLY=0
#export DATA_DIR_PREFIX="/p/cscratch/fs/hai_mlperf/deepcam_hdf5/"
CONFIG=configs/best_configs/config_DGXA100_2048GPU_BS2048_graph.sh
export SEED="$(date +%s)"

./start_training_run.sh -s booster -N $((TRAINING_INSTANCE_SIZE/4)) -c $CONFIG  -t 00:12:00
