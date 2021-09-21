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

hhai_dir="/p/project/jb_benchmark/MLPerf-1.0-combined/MLPerf/"
base_dir="${hhai_dir}benchmarks-closed/cosmoflow/"

export OUTPUT_ROOT="/p/project/jb_benchmark/MLPerf-1.0-combined/MLPerf/results"

export COSMOFLOW_DIR="${base_dir}/cosmoflow/"
# director for image: /workspace/cosmoflow/
#export CUDA_AVAILABLE_DEVICES="0,1,2,3"

SCRIPT_DIR="${base_dir}"
#"/p/project/jb_benchmark/MLPerf-1.0/mlperf-cosmoflow/"
SINGULARITY_FILE="/p/project/jb_benchmark/MLPerf-1.0/mlperf-cosmoflow/nvidia-cosmo-image.sif"

if [ -n "${CONFIG_FILE}" ]
  then
    export CONFIG_FILE="${SCRIPT_DIR}cosmoflow/configs/config_DGXA100_common.sh"
fi
echo "${CONFIG_FILE}"
cat "${CONFIG_FILE}"


srun "${SRUN_PARAMS[@]}" singularity exec --nv \
  --bind "${DATA_DIR_PREFIX}":/data,${SCRIPT_DIR},${OUTPUT_ROOT} ${SINGULARITY_FILE} \
    bash -c "\
      source ${CONFIG_FILE}; \
      bash run_and_time.sh"