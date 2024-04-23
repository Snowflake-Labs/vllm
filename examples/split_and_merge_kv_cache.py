import torch
from typing import Tuple


def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = 16 // kv_cache.element_size()
    num_blocks = kv_cache.shape[1]

    key_cache = kv_cache[0]
    key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                               -1, x)
    value_cache = kv_cache[1]
    value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
    return key_cache, value_cache


def merge_kv_cache(
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Assuming key_cache and value_cache have been reshaped to original flattened forms if needed
    x = 16 // key_cache.element_size()
    num_blocks = key_cache.shape[0]

    # Reshape to match the original kv_cache format before splitting
    key_cache = key_cache.view(num_blocks, -1)
    value_cache = value_cache.view(num_blocks, -1)
    return key_cache, value_cache


num_blocks = 100
block_size = 8
num_kv_heads = 16
head_size = 32

k_c = torch.arange(0, num_blocks * block_size * num_kv_heads * head_size)
key_cache = k_c.reshape((num_blocks, block_size * num_kv_heads * head_size))
value_cache = k_c.reshape((num_blocks, block_size * num_kv_heads * head_size))

kv_cache = torch.stack((key_cache, value_cache), dim=0)
new_key_cache, new_value_cache = split_kv_cache(kv_cache, num_kv_heads, head_size)

merged_key_cache, merged_value_cache = merge_kv_cache(new_key_cache, new_value_cache, num_kv_heads, head_size)

print(merged_key_cache==key_cache)
print(merged_value_cache==value_cache)





# attn_bias = BlockDiagonalCausalMask.from_seqlens(attn_metadata.prompt_lens)
# from xformers.ops.fmha.attn_bias import LocalAttentionFromBottomRightMask
# attn_bias = LocalAttentionFromBottomRightMask(
#     q_seqinfo=attn_bias.q_seqinfo,
#     k_seqinfo=attn_bias.k_seqinfo,
#     _batch_sizes=attn_bias._batch_sizes,
#     window_left=int(1e5),
#     window_right=int(1e5),
# )