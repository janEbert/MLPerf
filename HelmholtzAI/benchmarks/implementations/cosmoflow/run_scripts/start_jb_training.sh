#!/bin/bash
# This file is the first things to be done with srun

ml purge
#ml GCC OpenMPI
#cd /p/project/hai_mlperf/MLPerf-1.0/mlperf-deepcam/src/deepCam/run_scripts

SRUN_PARAMS=(
  --mpi            pspmix
  --cpu-bind       none
  #--label
)

export SLURM_CPU_BIND_USER_SET="none"

export USE_IME="${USE_IME:-0}"
# Whether to use HDF5 data.
export USE_H5=1
export READ_CHUNK_SIZE=64

# How many parallel trainings to run to test weak scaling
# (strong scaling has `INSTANCES=1`).
export INSTANCES=${INSTANCES:-1}

if ((USE_H5)); then
    if ((USE_IME)); then
        export HDF5_USE_FILE_LOCKING=FALSE
        export DATA_DIR_PREFIX="/p/cscratch/fs/hai_mlperf/cosmoflow"
    else
        export DATA_DIR_PREFIX="/p/scratch/hai_mlperf/cosmoflow"
    fi
else
    if ((USE_IME)); then
        export DATA_DIR_PREFIX="/p/cscratch/fs/hai_mlperf/cosmoUniverse_2019_05_4parE_tf_v2_numpy"
    else
        export DATA_DIR_PREFIX="/p/scratch/hai_mlperf/cosmoUniverse_2019_05_4parE_tf_v2_numpy"
    fi
fi

if ((USE_IME)); then
    find "$DATA_DIR_PREFIX"/train -type f -print0 \
        | xargs -n1 -P48 -0 ime-ctl --prestage
    find "$DATA_DIR_PREFIX"/validation -type f -print0 \
        | xargs -n1 -P48 -0 ime-ctl --prestage
fi

hhai_dir="/p/project/hai_mlperf/$USER/MLPerf/HelmholtzAI/"
base_dir="${hhai_dir}benchmarks/implementations/cosmoflow/"

export RESULTS_ROOT=${RESULTS_ROOT:-"/p/scratch/hai_mlperf/cosmoflow/results"}
mkdir -p "$RESULTS_ROOT"

export COSMOFLOW_DIR="${base_dir}cosmoflow/"
# director for image: /workspace/cosmoflow/
#export CUDA_AVAILABLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_IB_TIMEOUT=20

SCRIPT_DIR="${base_dir}"
#"/p/project/hai_mlperf/MLPerf-1.0/mlperf-cosmoflow/"
# APPTAINER_FILE="/p/project/hai_mlperf/MLPerf-1.0/mlperf-cosmoflow/nvidia-cosmo-image.sif"
APPTAINER_FILE=/p/project/hai_mlperf/jb_benchmark/nvidia_singularity_images/nvidia_cosmoflow_21.09_h5py_update.sif

if [ -z "${CONFIG_FILE}" ]; then
    export CONFIG_FILE="${SCRIPT_DIR}cosmoflow/configs/config_DGXA100_common.sh"
fi
echo "${CONFIG_FILE}"
cat "${CONFIG_FILE}"

srun "${SRUN_PARAMS[@]}" apptainer run --nv \
    --bind "${DATA_DIR_PREFIX}":/data,/tmp:"${STAGING_AREA}","${RESULTS_ROOT}":/results,"${SCRIPT_DIR}","${OUTPUT_ROOT}" \
    ${APPTAINER_FILE} \
    bash -c "\
      PMIX_SECURITY_MODE=native; \
      HOME=''; \
      source ${CONFIG_FILE}; \
      export DATA_SHARD_MULTIPLIER=\$((\$DATA_SHARD_MULTIPLIER * 2)); \
      export DGXNNODES=$SLURM_NNODES; \
      export DGXNGPU=4; \
      bash run_and_time.sh"
