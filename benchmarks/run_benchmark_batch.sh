python3 benchmark_batch.py \
    --warmup 1 \
    -n 1,4,8,16,32,64,128 \
    -l 2048 \
    --max_new_tokens 256 \
    -tp 8 \
    --framework vllm \
    --quantization deepspeedfp \
    --model /data-fast/yak2b-instruct-tp8 \
    --load_format sharded_state 

