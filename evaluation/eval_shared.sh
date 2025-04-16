#!/bin/bash

echo $SLURM_JOB_ACCOUNT

export BASE=/ocean/projects/$SLURM_JOB_ACCOUNT/$USER
export HF_HOME=$BASE/.hf

module load anaconda3
module load cuda
module load nvhpc

cd $BASE
conda activate $BASE/conda_cuda

DATASET=loogle
DATA_DIR=shortdep_qa
MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
PRESS=observed_attention_press

run() {
  echo "Evaluating press $PRESS with compression ratio $1 and $2-bit quantization"
  python3 evaluate.py --device "cuda:0" --dataset $DATASET --data_dir $DATA_DIR \
    --model $MODEL \
    --press $PRESS --compression_ratio $1 --quanto_bits $2 \
    > "results/$PRESS-$1-$2-stdout.log" \
    2> "results/$PRESS-$1-$2-stdout.log"&
}

run 0.5 8
