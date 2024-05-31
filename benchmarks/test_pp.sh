python benchmark_online.py \
        --prompt_length 2048 \
        --max_new_tokens 256 \
        --client_num 1024 \
        -qps 32,16,8,4,2,1 \
        --framework vllm \
        --model /data-fast/yak2b-25B-500B-phase3-instruct-v30-hf-gqa-vllm-fp8-tp8-test
#        --model /data-fast/yak2b-25B-500B-phase3-instruct-v30-hf-gqa
#        --model /data-fast/yak2c-hf
