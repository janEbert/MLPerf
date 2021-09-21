#!/bin/bash
# This file is the first things to be done with srun

ml purge
#ml GCC OpenMPI
#cd /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/src/deepCam/run_scripts

SRUN_PARAMS=(
  --mpi            pspmix
#  --cpu-bind       none
  #--label
)

export SLURM_CPU_BIND_USER_SET="ldoms"

export DATA_DIR_PREFIX="/p/scratch/jb_benchmark/cosmoUniverse_2019_05_4parE_tf_v2_numpy"

export OUTPUT_ROOT="/p/project/jb_benchmark/MLPerf-1.0/mlperf-cosmoflow/run-logs"

export DEEPCAM_DIR="/opt/deepCam/"
#export CUDA_AVAILABLE_DEVICES="0,1,2,3"

SCRIPT_DIR="/p/project/jb_benchmark/MLPerf-1.0/mlperf-cosmoflow/"
SINGULARITY_FILE="/p/project/jb_benchmark/MLPerf-1.0/mlperf-cosmoflow/nvidia-cosmo-image.sif"

CONFIG_FILE="${SCRIPT_DIR}cosmoflow/configs/config_DGXA100_common.sh"

srun "${SRUN_PARAMS[@]}" singularity exec --nv \
  --bind "${DATA_DIR_PREFIX}":/data ${SINGULARITY_FILE} \
    bash -c "\
      source ${CONFIG_FILE}; \
      export SLURM_CPU_BIND_USER_SET=\"none\"; \
      bash run_and_time.sh"
