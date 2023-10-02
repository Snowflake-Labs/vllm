"""
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
"""

import matplotlib.pyplot as plt
import numpy as np

MODELS_TO_NAME = {
    "meta-llama/Llama-2-7b-hf": "LLaMA-2 7B",
    "TheBloke/Llama-2-7B-AWQ": "LLaMA-2 7B\n(AWQ INT4)",
    "mistralai/Mistral-7B-v0.1": "Mistral 7B",
    "tiiuae/falcon-7b": "Falcon 7B",
    "mosaicml/mpt-7b": "MPT 7B",
    "EleutherAI/gpt-j-6b": "GPT-J 6B",
    "bigscience/bloom-7b1": "BLOOM 7B",
    "codellama/CodeLlama-7b-hf": "CodeLlama 7B",
}

HF = {
    "meta-llama/Llama-2-7b-hf": 53.07,
    "TheBloke/Llama-2-7B-AWQ": 0.0,
    "mistralai/Mistral-7B-v0.1": 59.98,
    "tiiuae/falcon-7b": 47.87,
    "mosaicml/mpt-7b": 47.70,
    "EleutherAI/gpt-j-6b": 41.47,
    "bigscience/bloom-7b1": 41.13,
    "codellama/CodeLlama-7b-hf": 39.93,
}

VLLM = {
    "meta-llama/Llama-2-7b-hf": 53.07,
    "TheBloke/Llama-2-7B-AWQ": 52.82,
    "mistralai/Mistral-7B-v0.1": 60.49,
    "tiiuae/falcon-7b": 47.95,
    "mosaicml/mpt-7b": 47.44,
    "EleutherAI/gpt-j-6b": 41.30,
    "bigscience/bloom-7b1": 41.13,
    "codellama/CodeLlama-7b-hf": 42.66,
}

# Draw a bar chart
fig, ax = plt.subplots(figsize=(12, 5))
width = 0.35
x = np.arange(len(HF))
ax.bar(x - width/2, HF.values(), width, label="HF", color="#fdb515ff")
# Add text on top of the bars
for i, v in enumerate(HF.values()):
    ax.text(i - width/2, v + 1, f"{v:.1f}", color="black", ha="center", fontsize=12)
ax.bar(x + width/2, VLLM.values(), width, label="vLLM", color="#30a2ffff")
# Add text on top of the bars
for i, v in enumerate(VLLM.values()):
    ax.text(i + width/2, v + 1, f"{v:.1f}", color="black", ha="center", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([MODELS_TO_NAME[m] for m in HF.keys()], fontsize=13)
ax.set_ylabel("Accuracy", fontsize=15)
ax.set_ylim(0, 70)
# Put legend on top of the plot
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=15)
plt.tight_layout()
plt.savefig("acc_plot.png")
