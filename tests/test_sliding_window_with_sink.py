import torch.distributed
import pytest
import torch
from unittest.mock import MagicMock

# from vllm.attention.backends.sink_rotations import SinkAttentionRotaryImpl


from typing import Callable
from unittest.mock import MagicMock

CONTEXT_LEN_1 = list(range(1, 15))


class BackedUpSink:
    def __init__(self):
        self.sink_key_cache = []
        self.sink_blocks = []
        self.is_empty = []

    def append(self, sink_key_cache: torch.Tensor, sink_blocks: torch.Tensor):
        self.sink_key_cache.append(sink_key_cache)
        self.sink_blocks.append(sink_blocks)
        self.is_empty.append(False)

    def append_empty(self):
        self.sink_key_cache.append(torch.empty(0))
        self.sink_blocks.append(torch.empty(0, dtype=torch.long))
        self.is_empty.append(True)

    def __iter__(self):
        for batch_i, (backup, blocks) in enumerate(
            zip(self.sink_key_cache, self.sink_blocks)
        ):
            if not self.is_empty[batch_i]:
                yield batch_i, backup, blocks

    @property
    def all_empty(self) -> bool:
        return all(self.is_empty)


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
        for _, prefix_sinks_pre_roll, sink_blocks in backed_up_sink:
            key_cache[sink_blocks] = prefix_sinks_pre_roll

    def process_decode_metadata(
        self, attn_metadata, key_cache: torch.Tensor, rotary_emb: Callable
    ) -> BackedUpSink:
        decode_meta = attn_metadata.decode_metadata
        backed_up_sink = BackedUpSink()

        if self.sink_size > 0:
            self._prepare_sink_rotation(decode_meta, key_cache, backed_up_sink)
            self._rotate_sinks(decode_meta, key_cache, rotary_emb, backed_up_sink)
        return backed_up_sink

    def _prepare_sink_rotation(
        self, decode_meta, key_cache: torch.Tensor, backed_up_sink: BackedUpSink
    ):
        """Prepare and return backup of sink positions for potential restoration."""
        for batch_i, batch_context_len in enumerate(decode_meta.context_lens):
            num_total_tokens_evicted = batch_context_len - self.cache_size
            if num_total_tokens_evicted > 0:
                self._backup_sink(
                    batch_i, batch_context_len, decode_meta, key_cache, backed_up_sink
                )
            else:
                backed_up_sink.append_empty()

    def _backup_sink(
        self,
        batch_i: int,
        batch_context_len: int,
        decode_meta,
        key_cache: torch.Tensor,
        backed_up_sink: BackedUpSink,
    ) -> None:
        num_sinks_current = (
            min(self.sink_size, batch_context_len) // key_cache.shape[-2]
        )
        sink_blocks = decode_meta.block_tables[batch_i, :num_sinks_current]
        sink_key_cache = torch.index_select(key_cache, index=sink_blocks, dim=0)
        backed_up_sink.append(sink_key_cache.clone(), sink_blocks)

    def _rotate_sinks(
        self,
        decode_meta: MagicMock,
        key_cache: torch.Tensor,
        rotary_emb: MagicMock,
        backed_up_sink: BackedUpSink,
    ):
        """Perform rotation on sinks and put it rotated in the cache."""
        for batch_i, backup, blocks in backed_up_sink:
            self._rotate_sink_positions(
                backup, blocks, decode_meta, key_cache, rotary_emb, batch_i
            )

    def _rotate_sink_positions(
        self,
        backup: torch.Tensor,
        blocks: torch.Tensor,
        decode_meta,
        key_cache: torch.Tensor,
        rotary_emb: MagicMock,
        batch_i: int,
    ) -> None:
        # get rotations angles
        num_total_tokens_evicted = self._calculate_evictions(decode_meta, batch_i)
        rotation_positions = (
            torch.ones(1, self.sink_size).to(key_cache.device)
            * num_total_tokens_evicted
        ).to(int)

        # rotate
        sink_to_rotate = self._format_key_cache_to_rotation(backup)
        dummy_query = torch.zeros_like(sink_to_rotate).to(key_cache.device)
        _, rotated_sinks = rotary_emb(rotation_positions, dummy_query, sink_to_rotate)

        # Restore rotated sinks into the original position in the cache
        key_cache[blocks[0]] = rotated_sinks.view(
            rotated_sinks.shape[0],  # fixme: only the first block
            self.num_kv_heads,
            self.head_size // 8,
            8,
        ).permute(1, 2, 0, 3)

    def _format_key_cache_to_rotation(self, x):
        # in: bs,  num_kv_heads, self.head_size/8, 16, 8
        return x.permute(3, 0, 1, 2, 4).reshape(self.sink_size, -1)

    def _calculate_evictions(self, decode_meta: MagicMock, batch_i: int):
        return max(decode_meta.context_lens[batch_i] - self.cache_size, 0)


@pytest.fixture
def setup_environment_for_sink(request):
    # Configure a mock environment for testing
    MAX_BLOCK_PER_ONE_EL = 15
    batch_size = 2
    num_blocks = MAX_BLOCK_PER_ONE_EL * batch_size
    # key shapes
    num_kv_heads = 5
    head_size = 128
    block_size = 3
    # cache persistency
    sliding_window = 6
    sink_size = 3

    context_len1 = request.param
    context_len2 = 1 + (context_len1 + 10) % MAX_BLOCK_PER_ONE_EL

    # Creating dummy tensors for key_cache and dummy_query (assuming dtype=torch.float32 for simplicity)
    key_cache = torch.rand(num_blocks, num_kv_heads, head_size // 8, block_size, 8)
    decode_meta = MagicMock()  # Simulating the metadata
    decode_meta.block_tables = (
        torch.arange(num_blocks).__reversed__().reshape(batch_size, -1)
    )
    decode_meta.context_lens = torch.LongTensor((context_len1, context_len2))
    attn_metadata = MagicMock()
    attn_metadata.decode_metadata = decode_meta
    sink_attn_obj = SinkAttentionRotaryImpl(
        sink_size, sliding_window, num_kv_heads, head_size
    )

    def _fake_rotate(x, pos):
        x.add_(pos[0, :, None].repeat(1, x.shape[-1]))
        return x

    rotary_emb = MagicMock(
        side_effect=lambda positions, dummy_query, sink_key: (
            _fake_rotate(dummy_query, positions),
            _fake_rotate(sink_key, positions),
        )
    )
    return sink_attn_obj, key_cache, attn_metadata, rotary_emb


@pytest.mark.parametrize("setup_environment_for_sink", CONTEXT_LEN_1, indirect=True)
def test_process_decode_metadata_and_restore(setup_environment_for_sink):
    sink_attn_obj, key_cache, attn_metadata, rotary_emb = setup_environment_for_sink
    initial_key_cache = key_cache.clone()
    # Run the method under test
    backed_up_sink = sink_attn_obj.process_decode_metadata(
        attn_metadata, key_cache, rotary_emb
    )
    # key cache should be updated
    if not backed_up_sink.all_empty:
        assert torch.max(key_cache - initial_key_cache) > 0
        assert torch.min(key_cache - initial_key_cache) == 0
        assert torch.all(
            torch.logical_or(
                torch.isclose(
                    ((key_cache - initial_key_cache) % 1.0),
                    torch.ones_like(initial_key_cache),
                ),
                torch.isclose(
                    ((key_cache - initial_key_cache) % 1.0),
                    torch.zeros_like(initial_key_cache),
                ),
            )
        )
        assert rotary_emb.call_count == len(backed_up_sink.is_empty) - sum(
            backed_up_sink.is_empty
        )  # Assuming it should be called once in this scenario

    # Check if rotary_emb was called, indicating rotation occurred
    sink_attn_obj.restore_cache_from_backup(key_cache, backed_up_sink)
    assert torch.equal(key_cache, initial_key_cache)