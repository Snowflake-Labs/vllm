from typing import List, Optional

from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper
import torch

from vllm._C import cache_ops
from vllm._C import ops
from vllm.model_executor.input_metadata import InputMetadata


class FlashInferImpl:

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        # Refer to
        # https://github.com/flashinfer-ai/flashinfer/blob/3d55c71a62052c590c130897d3a3db49b14fcc34/include/flashinfer/utils.cuh#L157
        return [64, 128, 256]

    @staticmethod
    def reshape_and_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> None:
        cache_ops.reshape_and_cache_flash(
            key,
            value,
            kv_cache,
            input_metadata.slot_mapping.flatten(),
            input_metadata.kv_cache_dtype,
        )

    @staticmethod
    def decode(
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        decode_wrapper: BatchDecodeWithPagedKVCacheWrapper,
    ) -> torch.Tensor:
        # 1. KV cache should be a single tensor, and change the layout.
        # 2. kv_page_indptr and kv_last_page_len should be prepared.
        # 3. decode_wrapper.begin_forward() should be executed before the model forward pass.
        return decode_wrapper.forward(query.contiguous(), kv_cache)

    @staticmethod
    def append(
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper,
    ) -> torch.Tensor:
        return prefill_wrapper.forward(query, kv_cache, causal=True)
