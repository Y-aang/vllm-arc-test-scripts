#!/bin/bash 

# python profile_vllm_server.py --port=8000 --temperature=0 --data_path=preprocess_data/shareGPT.json --stream --surplus_prompts_num=50 --use_burstgpt --prompt_num=50 --scale=1.2344107085 --burstgpt_path=../data/BurstGPT_1.csv

python profile_vllm_server.py \
    --port=8000 \
    --temperature=0 \
    --data_path=/home/shenyang/tests/burst/qwen-bailian-usagetraces-anon/processed.pkl \
    --stream \
    --surplus_prompts_num=3000 \
    --use_burstgpt \
    --prompt_num=3000 \
    --scale=1.0 \
    --burstgpt_path=../data/BurstGPT_1.csv \
    --workload=wild \