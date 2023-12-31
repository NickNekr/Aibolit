#!/bin/bash

base_model="sberbank-ai/rugpt3small_based_on_gpt2"
output_dir="./aibolit"

python3 pretrain_transformers.py \
  --output_dir=${output_dir} \
  --model_type=gpt2 \
  --model_name_or_path=${base_model} \
  --do_train \
  --train_data_file=data/train.txt \
  --do_eval \
  --eval_data_file=data/valid.txt \
  --per_gpu_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 5 \
  --block_size 384 \
  --overwrite_output_dir
