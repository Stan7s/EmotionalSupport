CUDA_VISIBLE_DEVICES=0,1 /afs/crc.nd.edu/user/l/ltong2/.conda/envs/bart/bin/python3.6 \
    -m torch.distributed.launch --nproc_per_node=2 finetune.py \
    --data_dir data/toy_m \
    --model_name_or_path facebook/bart-base \
    --output_dir finetuned_esp/toy_m \
    --num_train_epochs 50 \
    --max_source_length 768 \
    --max_target_length 32 \
    --val_max_target_length 32 \
    --test_max_target_length 32 \
    --learning_rate 1e-4 \
    --fp16 \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --load_best_model_at_end \
    --overwrite_output_dir \
    # --dataloader_num_workers 6 
    # --evaluate_during_training \
    # --prediction_loss_only \
    # --n_val 1000 \
    # "$@"
