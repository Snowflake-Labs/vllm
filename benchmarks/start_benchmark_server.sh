python -m vllm.entrypoints.api_server \
	--model /data-fast/yak2b-instruct \
	--swap-space 16 \
	--disable-log-requests \
	--tensor-parallel-size 8 \
	--pipeline-parallel-size 2 \
    --enforce-eager \
	--quantization deepspeedfp
