# redial
cd src
# cp -r data/inspired/. data/inspired_gen/
# cd data/inspired_gen
# python merge.py --gen_file_prefix prompt-for-conv-inspired

# cd ../..

export CUDA_VISIBLE_DEVICES=5

CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 train_rec.py \
    --dataset Redial_c2_copy_new \
    --tokenizer microsoft/DialoGPT-small \
    --model microsoft/DialoGPT-small \
    --text_tokenizer roberta-base \
    --text_encoder roberta-base \
    --n_prefix_rec 10 \
    --prompt_encoder ./pre-trained-prompt-vricr/best \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 530   \
    --context_max_length 200 \
    --prompt_max_length 200 \
    --entity_max_length 32 \
    --learning_rate 1e-4 \
    --output_dir ./prompt-for-rec-vricr \
    --seed 22