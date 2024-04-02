python benchmark_throughput.py \
    --dataset /shared/users/hao/project/vllm/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model /shared/finetuning/outputs/checkpoint/hf_ckpts/hao_ckpt/700M_tulu_yak_tp1 \
    --tensor-parallel-size 8 
    --quantization "yq" \
