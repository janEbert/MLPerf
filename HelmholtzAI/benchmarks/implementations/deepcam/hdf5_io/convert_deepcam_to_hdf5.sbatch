#!/usr/bin/env bash

#SBATCH --nodes=4
#SBATCH --tasks-per-node=8
#SBATCH --gres=gpu:0
#SBATCH --time=23:59:59
#SBATCH --partition=booster
#SBATCH --account=hai_mlperf

ml purge
ml Stages/2022 GCC/11.2.0 OpenMPI/4.1.2 Python/3.9.6 HDF5/1.12.1 mpi4py/3.1.3 h5py/3.5.0

srun python convert_deepcam_to_hdf5.py
