#!/bin/bash

#model='ai-forever/ruGPT-3.5-13B'
#python3 generate_transformers.py \
#  --model_type=gpt2 \
#  --model_name_or_path=${model} \
#  --repetition_penalty=1.1 \
#  --k=40 \
#  --p=0.85 \
#  --length=256 \
#  --stop_token='<|endoftext|>' \
#  --load_in_8bit

model="./aibolit"
python3 generate_transformers.py \
  --model_type=gpt2 \
  --model_name_or_path=${model} \
  --repetition_penalty=1.2 \
  --k=5 \
  --p=0.95 \
  --length=100
