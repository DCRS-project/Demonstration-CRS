# cp -r data/inspired src/data/
# cd src/data/inspired
cd src
python data/inspired/process.py
# cd ../../

# cp -r data/redial src/data/
# cd src
# python data/redial /process.py

CUDA_VISIBLE_DEVICES=1 accelerate launch --gpu_ids 1 train_pre_inspired.py \
    --dataset inspired \
    --tokenizer microsoft/DialoGPT-small \
    --model microsoft/DialoGPT-small \
    --text_tokenizer roberta-base \
    --text_encoder roberta-base \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --num_warmup_steps 168  \
    --max_length 200 \
    --prompt_max_length 200 \
    --entity_max_length 32 \
    --learning_rate 6e-4 \
    --output_dir ./pre-trained-prompt-inspired \
    --seed 1