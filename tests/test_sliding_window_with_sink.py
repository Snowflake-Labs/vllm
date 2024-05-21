import torch.distributed
import pytest
import torch
from unittest.mock import MagicMock

# from vllm.attention.backends.sink_rotations import SinkAttentionRotaryImpl


from typing import Callable, Optional
from unittest.mock import MagicMock

CONTEXT_LEN_1 = list(range(1, 15))




class BackedUpSink:
    def __init__(self):
        self.sink_key_cache: Optional[torch.Tensor] = None
        self.sink_blocks: Optional[torch.Tensor] = None

    def register(self, sink_key_cache: torch.Tensor, sink_blocks: torch.Tensor):
        """sink_blocks is flattened"""
        self.sink_key_cache = sink_key_cache
        self.sink_blocks = sink_blocks

    # def __iter__(self):
    #     for batch_i, (backup, blocks) in enumerate(
    #         zip(self.sink_key_cache, self.sink_blocks)
    #     ):
    #         yield batch_i, backup, blocks

    def __len__(self):
        return len(self.sink_blocks)


class SinkAttentionRotaryImpl(torch.nn.Module):
    def __init__(
        self,
        sink_size: int,
        sliding_window_size: int,
        num_kv_heads: int,
        head_size: int,
    ):
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
        # for _, backup, blocks in backed_up_sink:
        #     key_cache[blocks] = backup

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
        # for batch_i, batch_context_len in enumerate(decode_meta.context_lens):
        self._backup_sink(decode_meta, key_cache, backed_up_sink)

    def _backup_sink(
        self,
        decode_meta,
        key_cache: torch.Tensor,
        backed_up_sink: BackedUpSink,
    ) -> None:
        num_sinks_current = (
            self.sink_size // key_cache.shape[-2]
        )
        sink_blocks = decode_meta.block_tables[:, :num_sinks_current].flatten()
        # sink_key_cache = torch.index_select(key_cache, index=sink_blocks, dim=0)
        sink_key_cache = torch.index_select(key_cache, index=sink_blocks, dim=0)
        backed_up_sink.register(sink_key_cache.clone(), sink_blocks)

    def _rotate_sinks(
        self,
        key_cache: torch.Tensor,
        rotary_emb: MagicMock,
        backed_up_sink: BackedUpSink,
        positions: torch.Tensor,
    ):
        # get rotations angles
        num_evictions = self._calculate_evictions(positions)

        rotation_positions = (self._dummy_rotations * num_evictions[:, None]).to(int)

        # rotate
        bs = positions.shape[0]
        sink_to_rotate = backed_up_sink.sink_key_cache.permute(0, 3, 1, 2, 4).reshape(bs, self.sink_size, -1)
        # sink_to_rotate = self._format_key_cache_to_rotation(backed_up_sink.sink_key_cache)
        dummy_query = self._dummy_query.repeat(sink_to_rotate.shape)  # torch.zeros_like(sink_to_rotate).to(key_cache.device)
        _, rotated_sinks = rotary_emb(rotation_positions, dummy_query, sink_to_rotate)
        # dupa = sink_to_rotate.min(dim=-1).values
        # Put correctly rotated sinks into the original position in the cache
        key_cache[backed_up_sink.sink_blocks] = rotated_sinks.view(bs,
                                                                   self.sink_size,
                                                                   self.num_kv_heads,
                                                                   self.head_size // 8,
                                                                   8,
                                                                   ).permute(0, 2, 3, 1, 4)

    def _format_key_cache_to_rotation(self, x):
        # in: bs,  num_kv_heads, self.head_size/8, 16, 8
        return x.permute(3, 0, 1, 2, 4).reshape(self.sink_size, -1)

    def _calculate_evictions(self, positions: torch.Tensor):
        p_i = positions
        cs = self.cache_size
        diff = p_i - cs
        return torch.max(diff, self._cache_zeros)

    # Method to set the device of cache_size_gpu
    def to(self, device):
        self.cache_size = self.cache_size.to(device)
        self._cache_zeros = self._cache_zeros.to(device)
        self._dummy_rotations = self._dummy_rotations.to(device)
        self._dummy_query = self._dummy_query.to(device)
        return self  # Return self for method chaining


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
    sliding_window = 3
    sink_size = 3
    cache_size = sink_size + sliding_window

    context_len1 = request.param
    context_len2 = 1 + (context_len1 + 10) % MAX_BLOCK_PER_ONE_EL

    # Creating dummy tensors for key_cache and dummy_query (assuming dtype=torch.float32 for simplicity)
    key_cache = torch.rand(num_blocks, num_kv_heads, head_size // 8, block_size, 8)
    decode_meta = MagicMock()  # Simulating the metadata
    decode_meta.block_tables = (
        torch.arange(num_blocks).__reversed__().reshape(batch_size, -1)
    )
    context_lens = []
    for b_el in range(batch_size):
        context_lens.append(request.param + b_el*10 % MAX_BLOCK_PER_ONE_EL)
    positions = torch.LongTensor(context_lens)
    decode_meta.context_lens = torch.min(positions, torch.ones_like(positions) * cache_size).to(int)
    # decode_meta.context_lens = torch.LongTensor((min(context_len1, cache_size),
    #                                              min(cache_size, context_len2)))
    attn_metadata = MagicMock()
    attn_metadata.decode_metadata = decode_meta
    sink_attn_obj = SinkAttentionRotaryImpl(
        sink_size, sliding_window, num_kv_heads, head_size
    )

    def _fake_rotate(x, pos):
        x.add_(pos[:, :, None].repeat(1,1, x.shape[-1]))
        return x

    rotary_emb = MagicMock(
        side_effect=lambda positions, dummy_query, sink_key: (
            _fake_rotate(dummy_query, positions),
            _fake_rotate(sink_key, positions),
        )
    )
    return sink_attn_obj, key_cache, attn_metadata, rotary_emb, positions


@pytest.mark.parametrize("setup_environment_for_sink", CONTEXT_LEN_1, indirect=True)
def test_process_decode_metadata_and_restore(setup_environment_for_sink):
    sink_attn_obj, key_cache, attn_metadata, rotary_emb, positions = setup_environment_for_sink
    initial_key_cache = key_cache.clone()
    # Run the method under test
    backed_up_sink = sink_attn_obj.process_decode_metadata(
        attn_metadata, key_cache, rotary_emb, positions
    )
    # key cache should be updated
    if not torch.max(key_cache - initial_key_cache) >= 0:
        pass
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
    # assert rotary_emb.call_count == len(backed_up_sink)  # Assuming it should be called once for each elem in this scenario

    # Check if rotary_emb was called, indicating rotation occurred
    sink_attn_obj.restore_cache_from_backup(key_cache, backed_up_sink)
    assert torch.equal(key_cache, initial_key_cache)
