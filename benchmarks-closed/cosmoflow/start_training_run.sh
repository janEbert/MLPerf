#!/bin/bash

# hooray for stack overflow...
while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Launcher for training + timing for DeepCam on either HoreKa or Juwels Booster"
      echo " "
      echo "[options] application [arguments]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-s, --system              the HPC machine to use [horeka, booster]"
      echo "-N, --nnodes              number of nodes to compute on"
      echo "-t, --time                compute time limit"
      exit 0
      ;;
    -s)
      shift
      if test $# -gt 0; then
        export TRAINING_SYSTEM=$1
      else
        echo "must specify training system"
        exit 1
      fi
      shift
      ;;
    --system*)
      export TRAINING_SYSTEM=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
    -N)
      shift
      if test $# -gt 0; then
        export SLURM_NNODES=$1
      else
        echo "number of nodes must be positive"
        exit 1
      fi
      shift
      ;;
    --nnodes*)
      export SLURM_NNODES=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
    -t)
      shift
      if test $# -gt 0; then
        export TIMELIMIT=$1
      else
        echo "time limit must be greater than 0"
        exit 1
      fi
      shift
      ;;
    --time*)
      export TIMELIMIT=$1
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [ -z ${TIMELIMIT} ]; then TIMELIMIT="00:10:00"; fi

echo "Job time limit: "${TIMELIMIT}

SBATCH_PARAMS=(
  --nodes              "${SLURM_NNODES}"
  --tasks-per-node     "4"
  --time               "${TIMELIMIT}"
  --gres               "gpu:4"
  --job-name           "cosmoflow-mlperf"
  --time               "${TIMELIMIT}"
)

export TRAINING_SYSTEM="${TRAINING_SYSTEM}"

if [ "$TRAINING_SYSTEM" == "booster" ]
  then
    # JB
    export OUTPUT_DIR="/p/project/jb_benchmark/MLPerf-1.0/mlperf-cosmoflow/logs/"
    export OUTPUT_ROOT="/p/project/jb_benchmark/MLPerf-1.0/mlperf-cosmoflow/logs/"

    SBATCH_PARAMS+=(
      --partition     "booster"
      --output        "${OUTPUT_DIR}slurm-nodes-${SLURM_NNODES}-%j.out"
      --error         "${OUTPUT_DIR}slurm-nodes-${SLURM_NNODES}-%j.err"
      --account       "jb_benchmark"
    )
    echo "${TIMELIMIT}"
    sbatch "${SBATCH_PARAMS[@]}" start_jb_training.sh

elif [ "$TRAINING_SYSTEM" == "horeka" ]
  then
    export TRAIN_DATA_PREFIX="/hkfs/home/datasets/deepcam/"
    export OUTPUT_DIR="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/run-logs/"
    export OUTPUT_ROOT="/hkfs/work/workspace/scratch/qv2382-mlperf/mlperf-deepcam/run-logs/"

    SBATCH_PARAMS+=(
      --partition     "accelerated"
      --output        "${OUTPUT_DIR}slurm-nodes-${SLURM_NNODES}-%j.out"
      --error         "${OUTPUT_DIR}slurm-nodes-${SLURM_NNODES}-%j.err"
    )
    sbatch "${SBATCH_PARAMS[@]}" start_horeka_training.sh
else
  echo "must specify system that we are running on! give as first unnamed parameter"
  exit 128
fi
