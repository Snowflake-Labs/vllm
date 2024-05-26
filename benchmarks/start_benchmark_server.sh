python -m vllm.entrypoints.openai.api_server \
	--model /data-fast/yak2b-instruct-tp8 \
	--swap-space 16 \
	--tensor-parallel-size 8 \
	--pipeline-parallel-size 2 \
	--quantization deepspeedfp \
	--load-format sharded_state \
	--enforce-eager \
	--gpu-memory-utilization 0.8
