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
echo "Cached models"
huggingface-cli scan-cache

exit 0

mkdir -p $RESULTS_ROOT

run() {
  DEVICE=$1
  RATIO=$2
  QUANT=$3

  QFLAGS=${QUANT:+--quanto_bits $QUANT}
  QS=${QUANT:+q$QUANT}

  echo "Evaluating press $PRESS with compression ratio $1 ${QUANT:+$QUANT-bit quantization}"

  STDOUT="$RESULTS_ROOT/$PRESS-$1-$QS-stdout.log"
  STDERR="$RESULTS_ROOT/$PRESS-$1-$QS-stdout.log"

  echo "Piping output to $STDOUT and $STDERR"

  python3 evaluate.py \
    --device "cuda:$DEVICE" \
    --dataset $DATASET --data_dir $DATA_DIR \
    --model $MODEL \
    --press $PRESS \
    --compression_ratio $RATIO \
    $QFLAGS \
    --save_path $RESULTS_ROOT > $STDOUT > $STDERR &
}

mkdir -p results

run 0 0.75
run 1 0.0 4
run 2 0.8
run 3 0.2 4
run 4 0.85
run 5 0.4 4
run 6 0.875
run 7 0.5 4

wait
