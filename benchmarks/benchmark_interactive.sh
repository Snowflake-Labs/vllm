python3 benchmark_batch.py \
    --warmup 1 \
    -n 1,4,8,16,32 \
    -l 2048 \
    --max_new_tokens 256 \
    -tp 8 \
    --framework vllm \
    --quantization deepspeedfp \
    --load_format dummy \
    --model /data-fast/arctic-fp8-tp8
