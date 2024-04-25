"""Attention layer with xFormers and PagedAttention."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (AttentionBias,
                                         BlockDiagonalCausalMask,
                                         BlockDiagonalCausalLocalAttentionMask,
                                         LowerTriangularMaskWithTensorBias)
from vllm.attention.backends.causal_and_sink_attn import BlockDiagonalCausalLocalAttentionAndSinkMask
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataPerStage)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.logger import init_logger

logger = init_logger(__name__)


class XFormersBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["XFormersImpl"]:
        return XFormersImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "XFormersMetadata":
        return XFormersMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class XFormersMetadata(AttentionMetadataPerStage, PagedAttentionMetadata):
    """Metadata for XFormersbackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    # (batch_size,). The prompt length per sequence. None if it is a decoding.
    prompt_lens: Optional[List[int]]
    # prompt_lens stored as a tensor.
    prompt_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, subquery_len, and seqlen.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seqlen ----------------------|
    #                                   |- subquery_len -|

    # WARNING(sang): context_len has different definition depending on if it is
    # prefill vs decoding. When it is prefill, it doesn't include new tokens.
    # When it is for decoding, it includes a new token.

    # Maximum subquery length in the batch.
    max_subquery_len: Optional[int]
    # FIXME: It is for flash attn.
    # Maximum prompt length in the batch.
    max_prompt_len: Optional[int]
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    subquery_start_loc: Optional[torch.Tensor]
    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[AttentionBias]] = None


class XFormersImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|	
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:	
    |<----------------- num_decode_tokens ------------------>|	
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        sink_size: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        self.sink_size = sink_size
        self.cache_size = self.sliding_window + self.sink_size
        print(f"self.sliding_window = {self.sliding_window}")
        print(f"self.sink_size = {self.sink_size}")
        print(f"self.cache_size = {self.cache_size}")

        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata[XFormersMetadata],
        rotary_emb,
        kv_scale: float,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        num_tokens, hidden_size = query.shape
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                value_cache,
                                                attn_metadata.slot_mapping,
                                                attn_metadata.kv_cache_dtype,
                                                kv_scale)

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        print(f"num_prefill_tokens = {num_prefill_tokens}"
              f"num_decode_tokens = {num_decode_tokens}")

        assert key.shape[0] == num_prefill_tokens + num_decode_tokens
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_tokens]
        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if kv_cache is None or prefill_meta.block_tables.numel() == 0:
                # normal attention.
                out = self._run_memory_efficient_xformers_forward(
                    query, key, value, prefill_meta)
                assert out.shape == output[:num_prefill_tokens].shape
                output[:num_prefill_tokens] = out
            else:
                # prefix-enabled attention
                # TODO(Hai) this triton kernel has regression issue (broke) to
                # deal with different data types between KV and FP8 KV cache,
                # to be addressed separately.
                out = PagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.subquery_start_loc,
                    prefill_meta.prompt_lens_tensor,
                    prefill_meta.context_lens,
                    prefill_meta.max_subquery_len,
                    self.alibi_slopes,
                )
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out

        if decode_meta := attn_metadata.decode_metadata:
            print(f"decode meta,  num_prefill_tokens={num_prefill_tokens}, num_decode_tokens= {num_decode_tokens}")
            if self.sink_size is not None and self.sink_size > 0:
                self.uprotate_sink_positions(attn_metadata, decode_meta, key, key_cache, rotary_emb)

            output[num_prefill_tokens:] = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                decode_meta.block_tables,
                decode_meta.context_lens,
                decode_meta.max_context_len,
                attn_metadata.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                kv_scale,
            )

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)

    def uprotate_sink_positions(self, attn_metadata, decode_meta, key, key_cache, rotary_emb):
        for batch_i, batch_i_cl in enumerate(attn_metadata.decode_metadata.context_lens):
            # one single step, everything is in the cache
            # take the blocks that relate to sink tokens
            # rotate them, assigning positions at the end of the sliding window
            # q_len is num_tokens
            self._uprotate_sink_single_batch(batch_i, batch_i_cl, decode_meta, attn_metadata, key, key_cache, rotary_emb,
                                             self.sink_size, self.cache_size)

    @staticmethod
    def _uprotate_sink_single_batch(batch_i, batch_i_cl, decode_meta, attn_metadata, key, key_cache, rotary_emb, sink_size,
                                    cache_size):
        """supports  the case:
        [ ] AR decoding with 1 tokens to generate
            [ ] within the cache
            [ ] overflowing the cache
        """
        num_decode_tokens_this_pass = 1     # AR generating
        sink_block_size = 16
        assert batch_i_cl <= cache_size
        num_tokens_evicted_this_pass = int(batch_i_cl == cache_size)
        if num_tokens_evicted_this_pass:
            num_sinks_current = min(sink_size, batch_i_cl) // sink_block_size
            sink_blocks = decode_meta.block_tables[batch_i, :num_sinks_current]
            sink_key_cache = torch.index_select(key_cache, index=sink_blocks, dim=0)
            print(f"sink_key_cache.shape = {sink_key_cache.shape}")
            sink_key_to_roll = sink_key_cache.view(sink_size, -1)
            dummy_query_to_roll = torch.zeros_like(sink_key_to_roll).to(key.device)
            # we just evicted some tokens from cache, and we need to roll sink on their positions
            # find the additional rotations to apply on the sink
            # either we fit in the cache or need to evict one token and uprotate sink on this position.

            positions_one_bs = (torch.ones(1, num_sinks_current*sink_block_size).to(key.device) * num_tokens_evicted_this_pass).to(int)

            print(f"_uprotate_sink_single_batch, "
                  f"batch_i = {batch_i},  "
                  f"batch_i_cl = {batch_i_cl},  "
                  f"sink_blocks = {sink_blocks},  "
                  f"num_sinks_current = {num_sinks_current},  "
                  f"num_tokens_evicted_this_pass = {num_tokens_evicted_this_pass},  "
                  )
            _, _ = rotary_emb(positions_one_bs,
                              dummy_query_to_roll,
                              sink_key_to_roll)

    def _run_memory_efficient_xformers_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: XFormersMetadata,
    ) -> torch.Tensor:
        """Attention for 1D query of multiple prompts. Multiple prompt
        tokens are flattened in to `query` input.

        See https://facebookresearch.github.io/xformers/components/ops.html
        for API spec.

        Args:
            output: shape = [num_prefill_tokens, num_heads, head_size]
            query: shape = [num_prefill_tokens, num_heads, head_size]
            key: shape = [num_prefill_tokens, num_kv_heads, head_size]
            value: shape = [num_prefill_tokens, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        """
        original_query = query
        if self.num_kv_heads != self.num_heads:
            # GQA/MQA requires the shape [B, M, G, H, K].
            # Note that the output also has the same shape (which is different
            # from a spec from the doc).
            query = query.view(query.shape[0], self.num_kv_heads,
                               self.num_queries_per_kv, query.shape[-1])
            key = key[:, :,
                      None, :].expand(key.shape[0], self.num_kv_heads,
                                      self.num_queries_per_kv, key.shape[-1])
            value = value[:, :,
                          None, :].expand(value.shape[0], self.num_kv_heads,
                                          self.num_queries_per_kv,
                                          value.shape[-1])
        # Set attention bias if not provided. This typically happens at
        # the very attention layer of every iteration.
        # FIXME(woosuk): This is a hack.
        if attn_metadata.attn_bias is None:
            if self.alibi_slopes is None:
                if self.sink_size is not None:  # Create a custom tensor
                    from vllm.attention.backends.causal_and_sink_attn import _materialize_causal_mask_with_sink

                    attn_bias = _materialize_causal_mask_with_sink(
                            (query.shape[0], key.shape[0]),
                            dtype=key.dtype,
                            device=key.device,
                            window_size=self.sliding_window,
                            sink_size=self.sink_size,
                        )
                    # match the shape of B, H, Q, K
                    attn_bias = attn_bias[None, None, ...].expand(1, query.shape[1], query.shape[0], key.shape[0])
                # print(f"attn_metadata.prompt_lens = {attn_metadata.prompt_lens}")
                # attn_bias = BlockDiagonalCausalMask.from_seqlens(
                #     attn_metadata.prompt_lens)
                # if self.sliding_window is not None:
                #     # FIXME [MP]: hack for sink, as dense is still ok in the prompt
                #     print(f"attn_bias.q_seqinfo = {attn_bias.q_seqinfo}")
                #     print(f"attn_bias.k_seqinfo = {attn_bias.k_seqinfo}")
                #     print(f"attn_bias._batch_sizes = {attn_bias._batch_sizes}")
                #     print(f"self.sliding_window = {self.sliding_window}")
                #     print(f"self.sink_size = {self.sink_size}")
                #     attn_bias = BlockDiagonalCausalLocalAttentionAndSinkMask(
                #         q_seqinfo=attn_bias.q_seqinfo,
                #         k_seqinfo=attn_bias.k_seqinfo,
                #         _batch_sizes=attn_bias._batch_sizes,
                #         _window_size=self.sliding_window,
                #         _sink_size=self.sink_size,
                #     )
                #     # attn_bias = attn_bias.make_local_attention(int(1e5))
                attn_metadata.attn_bias = [attn_bias]
            else:
                attn_metadata.attn_bias = _make_alibi_bias(
                    self.alibi_slopes, self.num_kv_heads, query.dtype,
                    attn_metadata.prompt_lens)

        # No alibi slopes.
        # TODO(woosuk): Too many view operations. Let's try to reduce
        # them in the future for code readability.
        if self.alibi_slopes is None:
            # Add the batch dimension.
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            out = xops.memory_efficient_attention_forward(
                query,
                key,
                value,
                attn_bias=attn_metadata.attn_bias[0],
                p=0.0,
                scale=self.scale)
            return out.view_as(original_query)

        # Attention with alibi slopes.
        # FIXME(woosuk): Because xformers does not support dynamic sequence
        # lengths with custom attention bias, we process each prompt one by
        # one. This is inefficient, especially when we have many short prompts.
        output = torch.empty_like(original_query)
        start = 0
        for i, prompt_len in enumerate(attn_metadata.prompt_lens):
            end = start + prompt_len
            out = xops.memory_efficient_attention_forward(
                query[None, start:end],
                key[None, start:end],
                value[None, start:end],
                attn_bias=attn_metadata.attn_bias[i],
                p=0.0,
                scale=self.scale)
            # TODO(woosuk): Unnecessary copy. Optimize.
            output[start:end].copy_(out.view_as(original_query[start:end]))
            start += prompt_len
        return output


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    prompt_lens: List[int],
) -> LowerTriangularMaskWithTensorBias:
    attn_biases = []
    for prompt_len in prompt_lens:
        bias = torch.arange(prompt_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(prompt_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        # Calculate a matrix where each element represents ith element- jth
        # element.
        bias = bias[None, :] - bias[:, None]

        padded_len = (prompt_len + 7) // 8 * 8
        num_heads = alibi_slopes.shape[0]
        bias = torch.empty(
            1,  # batch size
            num_heads,
            prompt_len,
            padded_len,
            device=alibi_slopes.device,
            dtype=dtype,
        )[:, :, :, :prompt_len].copy_(bias)
        bias.mul_(alibi_slopes[:, None, None])
        if num_heads != num_kv_heads:
            bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
        attn_biases.append(LowerTriangularMaskWithTensorBias(bias))

    return attn_biases
