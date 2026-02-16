#!/bin/bash 

# python -m vllm.entrypoints.api_server --model /data2/share/llama/llama-2-7b-chat-hf
# /home/shenyang/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562

# python -m vllm.entrypoints.api_server --enable-prefix-caching --model /home/shenyang/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562
# /home/shenyang/tests/burst/qwen-bailian-usagetraces-anon/processed.pkl
# /home/shenyang/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
# Qwen/Qwen2.5-1.5B-Instruct  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

python -m vllm.entrypoints.api_server \
    --disable_sliding_window \
    --enable-prefix-caching \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --block_size 16 \
    --gpu-memory-utilization 0.233 \
    --max-model-len 4096 \
    --block-size 16