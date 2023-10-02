

### AI2 Reasoning Challenge (ARC)

We used [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) `arc_challenge` with 25 shots and reported `acc_norm`.

| Model | HF | vLLM |
|:-------|:-------:|:-------:|
| `meta-llama/Llama-2-7b-hf` | 53.07 | 53.07 |
| `TheBloke/Llama-2-7B-AWQ` | - | 52.82 |
| `mistralai/Mistral-7B-v0.1` | 59.98 | 60.49 |
| `tiiuae/falcon-7b` | 47.87 | 47.95 |
| `mosaicml/mpt-7b` | 47.70 | 47.44 |
| `EleutherAI/gpt-j-6b` | 41.47 | 41.30 |
| `bigscience/bloom-7b1` | 41.13 | 41.13 |
| `codellama/CodeLlama-7b-hf` | 39.93 | 42.66 |
