#!/bin/bash
# This file is the first things to be done with srun

ml purge
#ml GCC OpenMPI
#cd /p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/src/deepCam/run_scripts

SRUN_PARAMS=(
  --mpi            pspmix
  --cpu-bind       none
  #--label
)

export SLURM_CPU_BIND_USER_SET="none"

export USE_IME=0
# Whether to use HDF5 data.
export USE_H5=1
# Our HDF5 data is already pre-shuffled. If `USE_H5=1`, setting this
# to 1 has a large performance impact (either only on the staging part
# or on the whole run depending on `APPLY_PRESTAGE`).
export APPLY_PRESHUFFLE=$(if [ "$USE_H5" -ge 1 ]; then echo 0; else echo 1; fi)

# How many parallel trainings to run to test weak scaling
# (strong scaling has `INSTANCES=1`).
export INSTANCES=1

# Only apply prestaging when we
# 1. have only one parallel training run
# 2. have enough nodes to be able to support the memory requirements.
export APPLY_PRESTAGE=$(
    if [ "$INSTANCES" -gt 1 ]; then
        echo 0
    elif [ "$SLURM_NNODES" -ge 64 ]; then
        echo 1
    else
        echo 0
    fi
       )

if [[ ${USE_H5} -ge 1 ]]; then
    if [[ ${USE_IME} -ge 1 ]]; then
        export HDF5_USE_FILE_LOCKING=FALSE
        export DATA_DIR_PREFIX="/p/ime-scratch/fs/jb_benchmark/cosmoflow"
    else
        export DATA_DIR_PREFIX="/p/scratch/jb_benchmark/cosmoflow"
    fi
else
    if [[ ${USE_IME} -ge 1 ]]; then
        export DATA_DIR_PREFIX="/p/ime-scratch/fs/jb_benchmark/cosmoUniverse_2019_05_4parE_tf_v2_numpy"
    else
        export DATA_DIR_PREFIX="/p/scratch/jb_benchmark/cosmoUniverse_2019_05_4parE_tf_v2_numpy"
    fi
fi

hhai_dir="/p/project/jb_benchmark/MLPerf-1.0-combined/MLPerf/"
base_dir="${hhai_dir}benchmarks-closed/cosmoflow/"

export RESULTS_ROOT="/p/scratch/jb_benchmark/cosmoflow/results"
export OUTPUT_ROOT="/p/project/jb_benchmark/MLPerf-1.0-combined/MLPerf/results/cosmoflow/"

export COSMOFLOW_DIR="${base_dir}/cosmoflow/"
# director for image: /workspace/cosmoflow/
#export CUDA_AVAILABLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

SCRIPT_DIR="${base_dir}"
#"/p/project/jb_benchmark/MLPerf-1.0/mlperf-cosmoflow/"
# SINGULARITY_FILE="/p/project/jb_benchmark/MLPerf-1.0/mlperf-cosmoflow/nvidia-cosmo-image.sif"
SINGULARITY_FILE=/p/project/jb_benchmark/nvidia_singularity_images/nvidia_cosmoflow_21.09_h5py.sif

if [ -n "${CONFIG_FILE}" ]
  then
    export CONFIG_FILE="${SCRIPT_DIR}cosmoflow/configs/config_DGXA100_common.sh"
fi
echo "${CONFIG_FILE}"
cat "${CONFIG_FILE}"

srun "${SRUN_PARAMS[@]}" singularity exec --nv \
    --bind "${DATA_DIR_PREFIX}":/data,/tmp:"${STAGING_AREA}","${RESULTS_ROOT}":/results,${SCRIPT_DIR},${OUTPUT_ROOT} \
    ${SINGULARITY_FILE} \
    bash -c "\
      PMIX_SECURITY_MODE=native; \
      HOME=''; \
      source ${CONFIG_FILE}; \
      bash run_and_time.sh"
