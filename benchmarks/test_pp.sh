python benchmark_online.py \
	--prompt_length 2048 \
	--max_new_tokens 256 \
	-qps 1.0 \
	--framework vllm \
	--model /data-fast/yak2b-instruct
