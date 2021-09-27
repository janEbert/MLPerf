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

# Whether to use HDF5 data.
export USE_H5=1

if [[ ${USE_H5} -ge 1 ]]; then
    # We need to use prestaging with HDF5 data.
    export APPLY_PRESTAGE=1
    export DATA_DIR_PREFIX="/p/scratch/jb_benchmark/cosmoflow"
else
    export DATA_DIR_PREFIX="/p/scratch/jb_benchmark/cosmoUniverse_2019_05_4parE_tf_v2_numpy"
fi

hhai_dir="/p/project/jb_benchmark/MLPerf-1.0-combined/MLPerf/"
base_dir="${hhai_dir}benchmarks-closed/cosmoflow/"

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
  --bind "${DATA_DIR_PREFIX}":/data,${SCRIPT_DIR},${OUTPUT_ROOT} ${SINGULARITY_FILE} \
    bash -c "\
      PMIX_SECURITY_MODE=native; \
      HOME=''; \
      source ${CONFIG_FILE}; \
      bash run_and_time.sh"
