# Training with accelerate supports GPUs, TPUs and CPUs
# setup
> accelerate config
# run
> accelerate launch ./run_mlm.py \
    --model_name_or_path microsoft/deberta-v3-base \
    --tokenizer_name adalbertojunior/test-deberta \
    --train_file /home/m1/data/brwac-train.txt \
    --validation_file /home/m1/data/brwac-valid.txt \
    --max_seq_length 128 \
    --pad_to_max_length \
    --num_train_epochs 18 \
    --learning_rate 2e-4 \
    --output_dir output \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --save_steps=5000 \
    --cache_dir ./cache \
    --preprocessing_num_workers 32 \
    --num_warmup_steps 10000 \
    --gradient_accumulation_steps 16