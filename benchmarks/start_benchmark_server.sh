python -m vllm.entrypoints.api_server \
        --swap-space 16 \
        --tensor-parallel-size 8 \
        --enforce-eager \
	--gpu-memory-utilization 0.95 \
        --max-num-seqs 1024 \
        --max-num-batched-tokens 16384 \
        --enable-chunked-prefill \
        --disable-log-requests \
        --pipeline-parallel-size 3 \
        --model /data-fast/yak2c-hf \
        --pipeline-communication-method allgather \
#        --quantization deepspeedfp \
#        --model /data-fast/yak2b-25B-500B-phase3-instruct-v30-hf-gqa-vllm-fp8-tp8-test \
#        --load-format sharded_state \

