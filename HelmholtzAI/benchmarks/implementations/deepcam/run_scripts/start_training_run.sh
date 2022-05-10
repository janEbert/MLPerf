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
      echo "-N, --nodes               number of nodes to compute on"
      echo "-t, --time                compute time limit"
      echo "-c, --config              config file to use"
      exit 0
      ;;
    -s|--system) shift; export TRAINING_SYSTEM=$1; shift; ;;
    -N|--nodes) shift; export SLURM_NNODES=$1; shift; ;;
    -t|--time) shift; export TIMELIMIT=$1; shift; ;;
    -c|--config) shift; export CONFIG_FILE=$1; echo set config file; shift; ;;
    *) break; ;;
  esac
done

if [ -z "${TIMELIMIT}" ]; then TIMELIMIT="00:10:00"; fi

echo "Job time limit: "${TIMELIMIT}

SBATCH_PARAMS=(
  --nodes              "${SLURM_NNODES}"
  --tasks-per-node     "4"
  --time               "${TIMELIMIT}"
  --gres               "gpu:4"
  --job-name           "deepcam-mlperf"
  --time               "${TIMELIMIT}"
)

export TRAINING_SYSTEM="${TRAINING_SYSTEM}"

if [ "$TRAINING_SYSTEM" == "booster" ]
  then
    framework_and_version=pytorch1.10
    hhai_dir="/p/project/hai_mlperf/ebert1/MLPerf/HelmholtzAI/"

    n_total_gpus="$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))"
    weak_or_strong=$(
        if [ -z "${TRAINING_INSTANCE_SIZE}" ]; then
            n_instances=1
        else
            n_instances="$((n_total_gpus / TRAINING_INSTANCE_SIZE))"
        fi

        # Handle both cases where the variable is either not set, or
        # it is set but we don't actually have multiple instances.
        if [ "$n_instances" = 1 ]; then
            echo /strong
        else
            echo "${n_instances}x${TRAINING_INSTANCE_SIZE}/weak"
        fi)
    export OUTPUT_ROOT="${hhai_dir}results/juwelsbooster_gpu_n${n_total_gpus}_${framework_and_version}${weak_or_strong}/deepcam/"
    export OUTPUT_DIR="${OUTPUT_ROOT}"
    mkdir -p "$OUTPUT_DIR"

    SBATCH_PARAMS+=(
      --partition     "largebooster"
      --output        "${OUTPUT_DIR}slurm-deepcam-JB-N-${SLURM_NNODES}-%j.out"
      --error         "${OUTPUT_DIR}slurm-deepcam-JB-N-${SLURM_NNODES}-%j.err"
      --cpu-freq="high"
      --gpu-freq="high"
    )

    if [ -z $RESERVATION ]; then
      SBATCH_PARAMS+=(
        --account       "hai_cosmo"
      )
    else
      SBATCH_PARAMS+=(
        --account       "hai_mlperf"
        --reservation   "mlperf"
      )
    fi

    echo sbatch "${SBATCH_PARAMS[@]}" start_jb_training.sh
    sbatch "${SBATCH_PARAMS[@]}" start_jb_training.sh

elif [ "$TRAINING_SYSTEM" == "horeka" ]
  then
    hhai_dir="/hkfs/work/workspace/scratch/qv2382-mlperf-combined/MLPerf/"
    export OUTPUT_ROOT="${hhai_dir}results/deepcam/"
    export OUTPUT_DIR="${OUTPUT_ROOT}"
    mkdir -p "$OUTPUT_DIR"

    SBATCH_PARAMS+=(
      --partition     "accelerated"
      --output        "${OUTPUT_DIR}slurm-deepcam-HoreKa-N-${SLURM_NNODES}-%j.out"
      --error         "${OUTPUT_DIR}slurm-deepcam-HoreKa-N-${SLURM_NNODES}-%j.err"
      --exclude       "hkn[0518,0519,0533,0614,0625,0811]"
      --cpu-freq="high"
      --gpu-freq="high"
      --constraint="BEEOND"
      -A "hk-project-test-mlperf"
    )
    sbatch "${SBATCH_PARAMS[@]}" start_horeka_training.sh
else
  echo "must specify system that we are running on! give as first unnamed parameter"
  exit 128
fi
