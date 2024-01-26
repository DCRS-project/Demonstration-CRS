#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

export CUDA_VISIBLE_DEVICES=5

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/train_unsup_vricr.txt \
    --output_dir result/my-unsup-simcse-bert-base-uncased-vricr \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --learning_rate 3e-5 \
    --max_seq_length 512 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
