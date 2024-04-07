USE_DUMMY=True USE_FUSE=True python3 benchmark_batch.py \
    --warmup 1 \
    -n 1,2,4,8 \
    -l 2048 \
    --max_new_tokens 256 \
    -tp 8 \
    --framework vllm \
    --model /checkpoint/yak2b-25B-500B-phase2-instruct-v4-hf1