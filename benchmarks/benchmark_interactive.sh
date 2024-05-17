CUDA_LAUNCH_BLOCKING=1 USE_DUMMY=True USE_FUSE=True python3 benchmark_batch.py \
    --warmup 1 \
    -n 1,4,8,16 \ #,32,64,128 \
    -l 2048 \
    --max_new_tokens 256 \
    -tp 1 \
    --framework vllm \
    --model /data-fast/small-yak2c-long-seq-bookonly-eval-ckpt/conv1-fast-quant/2500/
