LR=$1
DATANAME=$2

MODEL_NAME_OR_PATH=/mnt/shared_data/users/t-wenhaoyu/nlg/dialogpt_model/medium
INIT_CHECKPOINT=/mnt/shared_data/users/t-wenhaoyu/nlg/dialogpt_model/medium/pytorch_model.bin

TRAIN_INPUT_FILE=/mnt/shared_data/users/t-wenhaoyu/nlg/full_short_0_-3/train.indexed.64len.db
EVAL_INPUT_FILE=/mnt/shared_data/users/t-wenhaoyu/nlg/full_short_0_-3/val.tsv
PRED_INPUT_FILE=/mnt/shared_data/users/t-wenhaoyu/nlg/full_short_0_-3/test.tsv
OUTPUT_DIR=/mnt/outputs/t-wenhaoyu/nlg_out/full_short_0_-3

python -m torch.distributed.launch --nproc_per_node=8 LSP_train.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --init_checkpoint $INIT_CHECKPOINT \
    --train_input_file $TRAIN_INPUT_FILE \
    --eval_input_file $EVAL_INPUT_FILE \
    --pred_input_file $PRED_INPUT_FILE \
    --output_dir $OUTPUT_DIR \
    --seed 42 \
    --max_seq_length 64 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --learning_rate 1e-4 \
    --num_optim_steps 320000 \
    --valid_step 2000 \
    --warmup_steps 2000 \
    --normalize_data true \
    --fp16 false \
    --lr_schedule noam \
    --loss_scale 0.0 \
    --no_token_id true \
    --pbar true
