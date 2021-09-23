#!/bin/bash
# This file is the first things to be done with srun

ml purge

SRUN_PARAMS=(
  --mpi            pspmix
  --cpu-bind       none
  #--label
)

export SLURM_CPU_BIND_USER_SET="none"

export DATA_DIR_PREFIX="/p/scratch/jb_benchmark/deepCam/"
export WIREUP_METHOD="nccl-slurm"
export SEED="0"

hhai_dir="/p/project/jb_benchmark/MLPerf-1.0-combined/MLPerf/"
base_dir="${hhai_dir}benchmarks-closed/deepcam/"
export DEEPCAM_DIR="${base_dir}image-src/"
#"/opt/deepCam/"

SCRIPT_DIR="${base_dir}run_scripts/"
SINGULARITY_FILE="${base_dir}docker/nvidia-optimized-image-2.sif"

export OUTPUT_ROOT="${hhai_dir}results/deepcam/"
export OUTPUT_DIR="${OUTPUT_ROOT}"

if [ -n "${CONFIG_FILE}" ]
  then
    export CONFIG_FILE="${SCRIPT_DIR}configs/best_configs/config_DGXA100_512GPU_BS1024_graph.sh"
fi
echo "${CONFIG_FILE}"
cat "${CONFIG_FILE}"

srun "${SRUN_PARAMS[@]}" singularity exec --nv \
  --bind "${DATA_DIR_PREFIX}",${SCRIPT_DIR},${OUTPUT_ROOT} ${SINGULARITY_FILE} \
    bash -c "\
      source ${CONFIG_FILE}; \
      bash run_and_time.sh"
