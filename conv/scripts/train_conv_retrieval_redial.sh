# train
cp -r data/redial src/data/
cd src
# python data/redial/process_mask.py

CUDA_VISIBLE_DEVICES=7 accelerate launch --gpu_ids 7 train_conv_retrieval_bart.py \
    --dataset redial \
    --tokenizer microsoft/DialoGPT-small \
    --model microsoft/DialoGPT-small \
    --text_tokenizer roberta-base \
    --text_encoder roberta-base \
    --n_prefix_conv 110 \
    --num_train_epochs 10 \
    --n_examples 3 \
    --mapping \
    --type_of_run training \
    --gradient_accumulation_steps 1 \
    --ignore_pad_token_for_loss \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_warmup_steps 6345 \
    --context_max_length 200 \
    --resp_max_length 80 \
    --prompt_max_length 50 \
    --entity_max_length 64 \
    --learning_rate 1e-4 \
    --output_dir ./prompt-conv-retrieval-bart-redial-42 \
    --seed 22