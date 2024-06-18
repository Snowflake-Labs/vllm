from typing import Callable, Optional

import torch


class BackedUpSink:
    def __init__(self) -> None:
        self.sink_key_cache: Optional[torch.Tensor] = None
        self.sink_blocks: Optional[torch.Tensor] = None

    def register(self, sink_key_cache: torch.Tensor, sink_blocks: torch.Tensor) -> None:
        """sink_blocks is flattened"""
        self.sink_key_cache = sink_key_cache
        self.sink_blocks = sink_blocks

    def __len__(self) -> int:
        return len(self.sink_blocks)


class SinkAttentionRotaryImpl(torch.nn.Module):
    def __init__(
        self,
        sink_size: int,
        sliding_window_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> None:
        super().__init__()
        self.sink_size = sink_size
        self.sliding_window_size = sliding_window_size
        self.cache_size = torch.Tensor([sliding_window_size + sink_size])
        self._cache_zeros = torch.Tensor([0])
        self._dummy_rotations = torch.ones(1, self.sink_size)
        self._dummy_query = torch.ones(1, 1)
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size

    def restore_cache_from_backup(
        self, key_cache: torch.Tensor, backed_up_sink: BackedUpSink
    ) -> None:
        key_cache[backed_up_sink.sink_blocks] = backed_up_sink.sink_key_cache

    def process_decode_metadata(
        self,
        attn_metadata,
        key_cache: torch.Tensor,
        rotary_emb: Callable,
        positions: torch.Tensor,
    ) -> BackedUpSink:
        assert rotary_emb is not None, f"Your model forward pass does not provide rotary_emb ({rotary_emb}). "
        assert positions is not None, f"Your model forward pass does not provide positions ({positions}). "
        decode_meta = attn_metadata.decode_metadata
        backed_up_sink = BackedUpSink()

        if self.sink_size > 0:
            self._backup_sink(decode_meta, key_cache, backed_up_sink)
            self._rotate_sinks(rotary_emb, key_cache, backed_up_sink, positions)
        return backed_up_sink

    def _backup_sink(
        self,
        decode_meta,
        key_cache: torch.Tensor,
        backed_up_sink: BackedUpSink,
    ) -> None:
        num_sinks_current = self.sink_size // key_cache.shape[-2]
        sink_blocks = decode_meta.block_tables[:, :num_sinks_current].flatten()
        sink_key_cache = torch.index_select(key_cache, index=sink_blocks, dim=0)
        backed_up_sink.register(sink_key_cache.clone(), sink_blocks)

    def _rotate_sinks(
        self,
        rotary_emb: Callable,
        key_cache: torch.Tensor,
        backed_up_sink: BackedUpSink,
        positions: torch.Tensor,
    ) -> None:
        # get rotations angles
        num_evictions = self._calculate_evictions(positions)
        bs = positions.shape[0]

        rotation_positions = (self._dummy_rotations * num_evictions[:, None]).to(int)

        # rotate
        sink_to_rotate = self._format_key_cache_to_rotation(
            backed_up_sink.sink_key_cache, bs
        )
        dummy_query = self._dummy_query.repeat(sink_to_rotate.shape)
        _, rotated_sinks = rotary_emb(rotation_positions, dummy_query, sink_to_rotate)
        # Put correctly rotated sinks into the original position in the cache
        key_cache[backed_up_sink.sink_blocks] = self._format_rotated_to_key_cache(
            rotated_sinks, bs
        )

    def _format_key_cache_to_rotation(self, x, bs) -> torch.Tensor:
        return x.permute(0, 3, 1, 2, 4).reshape(bs, self.sink_size, -1)

    def _format_rotated_to_key_cache(self, x, bs) -> torch.Tensor:
        return x.view(
            bs, self.sink_size, self.num_kv_heads, self.head_size // 8, 8
        ).permute(0, 2, 3, 1, 4)

    def _calculate_evictions(self, positions: torch.Tensor) -> torch.Tensor:
        return torch.max(positions - self.cache_size, self._cache_zeros)

    # Method to set the device of cache_size_gpu
    def to(self, device):
        self.cache_size = self.cache_size.to(device)
        self._cache_zeros = self._cache_zeros.to(device)
        self._dummy_rotations = self._dummy_rotations.to(device)
        self._dummy_query = self._dummy_query.to(device)
        return self  # Return self for method chaining
