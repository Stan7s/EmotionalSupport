#!/bin/bash
#$ -M ltong2@nd.edu
#$ -m abe
#$ -q gpu@qa-2080ti-007
#$ -pe smp 1
#$ -l gpu=4

MODEL_NAME_OR_PATH="models/large/"
INIT_CHECKPOINT="models/large/pytorch_model.bin"

TRAIN_INPUT_FILE="data/0518/full_short_0_-3/train.indexed.64len.db"
EVAL_INPUT_FILE="data/0518/full_short_0_-3/val.tsv"

OUTPUT_DIR="models/output_model/0518/full_short_0_-3/"

/afs/crc.nd.edu/user/l/ltong2/.conda/envs/LSP/bin/python -m torch.distributed.launch --nproc_per_node=4 /afs/crc/group/dmsquare/vol5/ltong2/DialoGPT/LSP_train.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --init_checkpoint $INIT_CHECKPOINT \
    --train_input_file $TRAIN_INPUT_FILE \
    --eval_input_file $EVAL_INPUT_FILE \
    --output_dir $OUTPUT_DIR \
    --seed 42 \
    --max_seq_length 64 \
    --train_batch_size 64 \
    --gradient_accumulation_steps 8 \
    --eval_batch_size 32 \
    --learning_rate 1e-5 \
    --num_optim_steps 400000 \
    --valid_step 5000 \
    --warmup_steps 10000 \
    --normalize_data true \
    --fp16 false \
    --lr_schedule noam \
    --loss_scale 0.0 \
    --no_token_id true \
    --pbar true
