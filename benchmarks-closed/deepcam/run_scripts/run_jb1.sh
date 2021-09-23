#!/bin/bash
export TRAINING_INSTANCE_SIZE=4
export STAGE_DIR_PREFIX=/tmp/deepcam
./start_training_run.sh -s booster -N 1 -c /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/src/deepCam/run_scripts/configs/best_configs/config_DGXA100_128GPU_BS128_graph.sh  -t 00:15:00
