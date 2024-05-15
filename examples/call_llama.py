import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.backends.xformers import XFormersMetadata
cfg = LlamaConfig(vocab_size = 350,
hidden_size = 256,
intermediate_size = 512,
num_hidden_layers = 2,
num_attention_heads = 32,
num_key_value_heads = None,
hidden_act = "silu",
max_position_embeddings = 2048,
initializer_range = 0.02,
rms_norm_eps = 1e-6,
use_cache = True,
pad_token_id = None,
bos_token_id = 1,
eos_token_id = 2,
pretraining_tp = 1,
tie_word_embeddings = False,
rope_theta = 10000.0,
rope_scaling = None,
attention_bias = False,
attention_dropout = 0.0,
)
num_blocks=1000
block_size = 16
head_size = cfg.hidden_size // cfg.num_attention_heads
x = 8
# key cache [num_blocks, num_heads, head_size/x, block_size, x]
# value_cache  [num_blocks, num_heads, head_size, block_size]
model = LlamaForCausalLM(cfg)
k_cache = torch.zeros((num_blocks, cfg.num_attention_heads, head_size/x, block_size, x))
v_cache = torch.zeros((num_blocks, cfg.num_attention_heads, head_size, block_size))

# prepare
seq_len = 10
input_ids = torch.arange(0, seq_len) + 50
positions = torch.arange(0, seq_len)
attn_metadata = XFormersMetadata(is_prompt=True,
prompt_lens=[seq_len],
prompt_lens_tensor=prompt_lens_tensor,
max_subquery_len=max_subquery_len,
max_context_len=None,
max_prompt_len=max_prompt_len,
subquery_start_loc=subquery_start_loc,
seq_start_loc=seq_start_loc,
context_lens=context_lens_tensor,
block_tables=block_tables,
use_cuda_graph=False)

model(
    input_ids = input_ids, #torch.Tensor,
    positions = positions, #torch.Tensor,
    kv_caches = [k_cache, v_cache ], #List[torch.Tensor],
    attn_metadata = attn_metadata, #: AttentionMetadata,
)
