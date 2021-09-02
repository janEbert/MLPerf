#!/bin/bash
# This file is the first things to be done with srun

ml purge

# pmi2 cray_shasta
SRUN_PARAMS=(
  --mpi="pmi2"
  #--cpu-bind="ldoms"
)

export SLURM_CPU_BIND_USER_SET="ldoms"

export DATA_DIR_PREFIX="/hkfs/home/dataset/datasets/deepcam_npy/"
export WIREUP_METHOD="nccl-slurm-pmi"
export SEED="0"


hhai_dir="/hkfs/work/workspace/scratch/qv2382-mlperf/"
base_dir="${hhai_dir}benchmarks-closed/deepcam/"
export DEEPCAM_DIR="${base_dir}image-src/"
#"/opt/deepCam/"

SCRIPT_DIR="${base_dir}run_scripts/"
SINGULARITY_FILE="${base_dir}docker/nvidia-optimized-torch.sif"

export OUTPUT_ROOT="${hhai_dir}results/"
export OUTPUT_DIR="${OUTPUT_ROOT}"

if [ -n "${CONFIG_FILE}" ]
  then
    export CONFIG_FILE="${SCRIPT_DIR}configs/best_configs/config_DGXA100_512GPU_BS1024_graph.sh"
fi
echo "${CONFIG_FILE}"
cat "${CONFIG_FILE}"

srun "${SRUN_PARAMS[@]}" singularity exec --nv \
  --bind "${DATA_DIR_PREFIX}",${SCRIPT_DIR},${OUTPUT_ROOT},${DEEPCAM_DIR} ${SINGULARITY_FILE} \
    bash -c "\
      source ${CONFIG_FILE}; \
      export SLURM_CPU_BIND_USER_SET=\"none\"; \
      bash run_and_time.sh"
