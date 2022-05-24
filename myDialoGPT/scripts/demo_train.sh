#!/bin/bash
#$ -M ltong2@nd.edu
#$ -m abe
#$ -q gpu@qa-2080ti-007
#$ -pe smp 1
#$ -l gpu=1

MODEL_NAME_OR_PATH="models/medium/"
INIT_CHECKPOINT="models/medium/pytorch_model.bin"

TRAIN_INPUT_FILE="data/dummy/train.indexed.128len.db"
EVAL_INPUT_FILE="data/dummy/val.tsv"
PRED_INPUT_FILE="data/dummy/pred.tsv"

OUTPUT_DIR="models/output_model/dummy/"

CUDA_VISIBLE_DEVICES=2 /afs/crc.nd.edu/user/l/ltong2/.conda/envs/LSP/bin/python LSP_train.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --init_checkpoint $INIT_CHECKPOINT \
    --train_input_file $TRAIN_INPUT_FILE \
    --eval_input_file $EVAL_INPUT_FILE \
    --pred_input_file $PRED_INPUT_FILE \
    --output_dir $OUTPUT_DIR \
    --seed 42 \
    --max_seq_length 128 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --eval_batch_size 8 \
    --learning_rate 1e-5 \
    --num_optim_steps 50 \
    --valid_step 10 \
    --warmup_steps 10 \
    --normalize_data true \
    --fp16 false \
    --lr_schedule noam \
    --loss_scale 0.0 \
    --no_token_id true \
    --pbar true


# /afs/crc.nd.edu/user/l/ltong2/.conda/envs/LSP/bin/python -m torch.distributed.launch --nproc_per_node=2 /afs/crc/group/dmsquare/vol5/ltong2/DialoGPT/LSP_train.py \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --init_checkpoint $INIT_CHECKPOINT \
#     --train_input_file $TRAIN_INPUT_FILE \
#     --eval_input_file $EVAL_INPUT_FILE \
#     --output_dir $OUTPUT_DIR \
#     --seed 42 \
#     --max_seq_length 128 \
#     --train_batch_size 8 \
#     --gradient_accumulation_steps 2 \
#     --eval_batch_size 8 \
#     --learning_rate 1e-5 \
#     --num_optim_steps 100 \
#     --valid_step 10 \
#     --warmup_steps 10 \
#     --normalize_data true \
#     --fp16 false \
#     --lr_schedule noam \
#     --loss_scale 0.0 \
#     --no_token_id true \
#     --pbar true
