#!/bin/bash
# This file is the first things to be done with srun

ml purge
#ml devel/cuda/11.4 
#ml compiler/gnu/11 mpi/openmpi/4.1

# pmi2 cray_shasta
SRUN_PARAMS=(
  --mpi="pmi2"
#  --label
  --cpu-bind="none"
#  --cpus-per-task="19"
  --unbuffered
)

export SLURM_CPU_BIND_USER_SET="ldoms"

#export DATA_DIR_PREFIX="/hkfs/home/dataset/datasets/deepcam_npy/"
export DATA_DIR_PREFIX="/hkfs/work/workspace/scratch/qv2382-mlperf_data/hdf5s/"
export DATA_CACHE_DIRECTORY="/mnt/odfs/${SLURM_JOB_ID}/stripe_8"
#export STAGE_DIR_PREFIX="${DATA_CACHE_DIRECTORY}"
export STAGE_DIR_PREFIX="/tmp/deepcam"

mkdir "${STAGE_DIR_PREFIX}"

export WIREUP_METHOD="nccl-slurm-pmi"
export SEED="0"


export HHAI_DIR="/hkfs/work/workspace/scratch/qv2382-mlperf-combined/MLPerf/"
base_dir="${HHAI_DIR}benchmarks-closed/deepcam/"
export DEEPCAM_DIR="${base_dir}image-src/"
#"/opt/deepCam/"

SCRIPT_DIR="${base_dir}run_scripts/"
SINGULARITY_FILE="${base_dir}docker/deepcam_optimized-21.09_2.sif"
# deepcam_optimized-21.09.sif"
# deepcam_optimized-21.09.sif"  
#nvidia-optimized-image-2.sif"

export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0

echo "${SINGULARITY_FILE}"

export OUTPUT_ROOT="${HHAI_DIR}results/deepcam/"
export OUTPUT_DIR="${OUTPUT_ROOT}"

echo "${CONFIG_FILE}"
cat "${CONFIG_FILE}"

ipaddr_dir="/etc/sysconfig/network-scripts/ifcfg-ib0"

srun "${SRUN_PARAMS[@]}" singularity exec --nv \
  --bind "${DATA_DIR_PREFIX}","${HHAI_DIR}","${OUTPUT_ROOT}","${DATA_CACHE_DIRECTORY}",/scratch,"${ipaddr_dir}",/tmp ${SINGULARITY_FILE} \
    bash -c "\
      source ${CONFIG_FILE}; \
      mkdir ${STAGE_DIR_PREFIX}; \
      export NCCL_DEBUG=INFO; \
      export SLURM_CPU_BIND_USER_SET=\"none\"; \
      bash run_and_time.sh"
