#!/bin/bash
#$ -M ltong2@nd.edu
#$ -m abe
#$ -q gpu@qa-2080ti-007
#$ -pe smp 1
#$ -l gpu=1

MODEL_NAME_OR_PATH="models/medium/"
CHECKPOINT="models/output_model/dummy/GPT2.1e-05.4.1gpu.2022-05-19185015/GP2-pretrain-step-10.pkl"

TEST_INPUT_FILE="data/dummy/pred.tsv"

CUDA_VISIBLE_DEVICES=0 /afs/crc.nd.edu/user/l/ltong2/.conda/envs/LSP/bin/python predict.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --load_checkpoint $CHECKPOINT \
    --pred_data $TEST_INPUT_FILE \
    --use_gpu

python eval_metrics.py --data "${CHECKPOINT%.*}.tsv"