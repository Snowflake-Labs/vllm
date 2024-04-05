USE_FUSE=True python benchmark_online.py --prompt_length 2048 \
    -tp 8 \
    --max_new_tokens 256 \
    -c 1024 \
    -qps 1.0 \
    --model NousResearch/Llama-2-70b-chat-hf \
    --framework vllm
