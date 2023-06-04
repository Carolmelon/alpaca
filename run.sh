CUDA_VISIBLE_DEVICES=1,4,5,6,7 torchrun --nproc_per_node=5 --master_port=12432 train.py \
    --model_name_or_path /home/lrs/test/stanford_alpaca/LLaMA/llama_7b/llama-7b \
    --data_path ./alpaca_data.json \
    --output_dir model_output_1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --fp16 True
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
    
#    --deepspeed ds_config.json \