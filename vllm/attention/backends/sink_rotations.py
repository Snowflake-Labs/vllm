from typing import Callable
from unittest.mock import MagicMock

import torch




class BackedUpSink:
    def __init__(self):
        self.sink_key_cache = []
        self.sink_blocks = []

    def register(self, sink_key_cache: torch.Tensor, sink_blocks: torch.Tensor):
        self.sink_key_cache.append(sink_key_cache)
        self.sink_blocks.append(sink_blocks)

    def __iter__(self):
        for batch_i, (backup, blocks) in enumerate(
            zip(self.sink_key_cache, self.sink_blocks)
        ):
            yield batch_i, backup, blocks

    def __len__(self):
        return len(self.sink_blocks)


class SinkAttentionRotaryImpl:
    def __init__(
        self,
        sink_size: int,
        sliding_window_size: int,
        num_kv_heads: int,
        head_size: int,
    ):
        self.sink_size = sink_size
        self.sliding_window_size = sliding_window_size
        self.cache_size = sliding_window_size + sink_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size

    def restore_cache_from_backup(
        self, key_cache: torch.Tensor, backed_up_sink: BackedUpSink
    ) -> None:
        for _, backup, blocks in backed_up_sink:
            key_cache[blocks] = backup

    def process_decode_metadata(
        self, attn_metadata, key_cache: torch.Tensor, rotary_emb: Callable, positions
    ) -> BackedUpSink:
        decode_meta = attn_metadata.decode_metadata
        backed_up_sink = BackedUpSink()

        if self.sink_size > 0:
            self._prepare_sink_rotation(decode_meta, key_cache, backed_up_sink, positions)
            self._rotate_sinks(key_cache, rotary_emb, backed_up_sink, positions)
        return backed_up_sink

    def _prepare_sink_rotation(
        self, decode_meta, key_cache: torch.Tensor, backed_up_sink: BackedUpSink, positions: torch.Tensor
    ):
        """Prepare and return backup of sink positions for potential restoration."""
        for batch_i, batch_context_len in enumerate(decode_meta.context_lens):
            self._backup_sink(
                    batch_i, decode_meta, key_cache, backed_up_sink
                )

    def _backup_sink(
        self,
        batch_i: int,
        decode_meta,
        key_cache: torch.Tensor,
        backed_up_sink: BackedUpSink,
    ) -> None:
        num_sinks_current = (
            self.sink_size // key_cache.shape[-2]
        )
        sink_blocks = decode_meta.block_tables[batch_i, :num_sinks_current]
        sink_key_cache = torch.index_select(key_cache, index=sink_blocks, dim=0)
        backed_up_sink.register(sink_key_cache.clone(), sink_blocks)

    def _rotate_sinks(
        self,
        key_cache: torch.Tensor,
        rotary_emb: MagicMock,
        backed_up_sink: BackedUpSink,
        positions: torch.Tensor,
    ):
        """Perform rotation on sinks and put it rotated in the cache."""
        for batch_i, backup, blocks in backed_up_sink:
            self._rotate_sink_positions(
                backup, blocks, key_cache, rotary_emb, self._calculate_evictions(positions, batch_i)
            )

    def _rotate_sink_positions(
        self,
        backup: torch.Tensor,
        blocks: torch.Tensor,
        key_cache: torch.Tensor,
        rotary_emb: Callable,
        num_total_tokens_evicted: int
    ) -> None:
        # get rotations angles
        rotation_positions = (
            torch.ones(1, self.sink_size).to(key_cache.device)
            * num_total_tokens_evicted
        ).to(int)

        # rotate
        sink_to_rotate = self._format_key_cache_to_rotation(backup)
        dummy_query = torch.zeros_like(sink_to_rotate).to(key_cache.device)
        _, rotated_sinks = rotary_emb(rotation_positions, dummy_query, sink_to_rotate)

        # Put correctly rotated sinks into the original position in the cache
        key_cache[blocks[0]] = rotated_sinks.view(
            rotated_sinks.shape[0],  # fixme: only the first block
            self.num_kv_heads,
            self.head_size // 8,
            8,
        ).permute(1, 2, 0, 3)

    def _format_key_cache_to_rotation(self, x):
        # in: bs,  num_kv_heads, self.head_size/8, 16, 8
        return x.permute(3, 0, 1, 2, 4).reshape(self.sink_size, -1)

    def _calculate_evictions(self, positions: torch.Tensor, batch_i: int):
        return max(positions[batch_i] - self.cache_size, 0)

