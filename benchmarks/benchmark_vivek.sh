USE_DUMMY=True USE_FUSE=True python benchmark_online.py \
    --prompt_length 2048 \
    -tp 8 \
    --max_new_tokens 256 \
    -c 1024 \
    -qps 1.0 \
    --model /checkpoint/yak2b-25B-500B-phase2-instruct-v4-hf1 \
    --framework vllm
