#!/bin/bash

echo $SLURM_JOB_ACCOUNT

export BASE=/ocean/projects/$SLURM_JOB_ACCOUNT/$USER
export HF_HOME=$BASE/.hf

module load anaconda3
module load cuda
module load nvhpc

cd $BASE
conda activate $BASE/conda_cuda

WORKING_DIR=$BASE/kvpress/evaluation

cd $WORKING_DIR

export TQDM_DISABLE=1
python3 multieval.py
