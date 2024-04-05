# Benchmarking vLLM

## Downloading the ShareGPT dataset

You can download the dataset by running:
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```



## Benchmark online
```bash
USE_FUSE=True python -m vllm.entrypoints.api_server --model=[YAK MODEL PATH] -tp=8 --trust-remote-code
```
then open another terminal and do:
```bash
bash benchmark_vivek.sh
```


## Benchmark offline
```bash
bash benchmark_interactive.sh
```