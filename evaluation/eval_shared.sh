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

DATASET=loogle
DATA_DIR=shortdep_qa
MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
PRESS=observed_attention_press

RESULTS_ROOT="/jet/home/$USER/results"

echo "Using HF_HOME=$HF_HOME"
mkdir -p $RESULTS_ROOT

run() {
  echo "Evaluating press $PRESS with compression ratio $1 and $2-bit quantization"

  STDOUT="$RESULTS_ROOT/$PRESS-$1-$2-stdout.log"
  STDERR="$RESULTS_ROOT/$PRESS-$1-$2-stdout.log"

  echo "Piping output to $STDOUT and $STDERR"


  python3 evaluate.py --device "cuda:0" --dataset $DATASET --data_dir $DATA_DIR \
    --model $MODEL \
    --press $PRESS \
    --compression_ratio $1 \
    --quanto_bits $2 \
    --save_dir $RESULTS_ROOT > $STDOUT > $STDERR &
}

mkdir -p results

run 0.5 8

wait
