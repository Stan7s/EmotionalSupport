INPUT_DIR=/mnt/shared_data/users/t-wenhaoyu/nlg
OUTPUT_DIR=/mnt/outputs/t-wenhaoyu/nlg_out

LR=$1
DATANAME=$2

python -m torch.distributed.launch --nproc_per_node=8 finetune.py \
    --data_dir ${INPUT_DIR}/${DATANAME} \
    --model_name_or_path facebook/bart-base \
    --output_dir ${OUTPUT_DIR}_${DATANAME}_${LR} \
    --num_train_epochs 50 \
    --max_source_length 768 \
    --max_target_length 64 \
    --val_max_target_length 64 \
    --test_max_target_length 64 \
    --learning_rate ${LR} \
    --fp16 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --predict_with_generate \
    --load_best_model_at_end \
    --overwrite_output_dir \
    # --dataloader_num_workers 6 
    # --evaluate_during_training \
    # --prediction_loss_only \
    # --n_val 1000 \
    # "$@"
