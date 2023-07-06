# echo $$ > finetune_llama_7b_reddit_one_example_1.pid
# nohup 

/home/kokil/ahmad/llms/envs/qlora_env/bin/python /home/kokil/ahmad/llms/projects/qlora_training/qlora/qlora.py \
    --model_name_or_path /home/kokil/ahmad/llms/ml-models/decapoda-research/llama-7b-hf \
    --output_dir /home/kokil/ahmad/llms/projects/qlora_training/qlora/output/llama-7b-hf-reddit-zero-example \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 100 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset /home/kokil/ahmad/llms/projects/qlora_training/hatespeech_data/dataset_dict/reddit/fsl_classification_dataset_zero_example \
    --dataset_format input-output \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1000 \
    --eval_steps 187 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0
     > script_output_6_1_3.log 2>&1 &