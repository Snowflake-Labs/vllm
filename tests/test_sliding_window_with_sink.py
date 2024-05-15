import torch.distributed
import pytest
import torch
from unittest.mock import MagicMock

# from vllm.attention.backends.sink_rotations import SinkAttentionRotaryImpl


from typing import Callable
from unittest.mock import MagicMock

import torch

# from vllm.attention import AttentionMetadata, AttentionMetadataPerStage


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
        for batch_i, (backup, blocks) in enumerate(zip(self.sink_key_cache, self.sink_blocks)):
            if not self.is_empty[batch_i]:
                yield batch_i, backup, blocks


class SinkAttentionRotaryImpl:
    def __init__(self, sink_size: int, sliding_window_size: int, num_kv_heads: int, head_size: int):
        self.sink_size = sink_size
        self.sliding_window_size = sliding_window_size
        self.cache_size = sliding_window_size + sink_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size

    def restore_cache_from_backup(self, key_cache: torch.Tensor, backed_up_sink: BackedUpSink) -> None:
        for _, prefix_sinks_pre_roll, sink_blocks in backed_up_sink:
            key_cache[sink_blocks] = prefix_sinks_pre_roll

    def process_decode_metadata(self, attn_metadata, key_cache: torch.Tensor, rotary_emb: Callable) -> BackedUpSink:
        decode_meta = attn_metadata.decode_metadata
        backed_up_sink = BackedUpSink()

        if self.sink_size > 0:
            self._prepare_sink_rotation(decode_meta, key_cache, backed_up_sink)
            self._rotate_sinks(decode_meta, key_cache, rotary_emb, backed_up_sink)
        return backed_up_sink

    def _prepare_sink_rotation(self, decode_meta, key_cache: torch.Tensor, backed_up_sink: BackedUpSink):
        """Prepare and return backup of sink positions for potential restoration."""
        for batch_i, batch_context_len in enumerate(decode_meta.context_lens):
            num_total_tokens_evicted = batch_context_len - self.cache_size
            if num_total_tokens_evicted > 0:
                self._backup_sink(batch_i, batch_context_len, decode_meta, key_cache, backed_up_sink)
            else:
                backed_up_sink.append_empty()

    def _backup_sink(self, batch_i: int, batch_context_len: int, decode_meta,
                    key_cache: torch.Tensor, backed_up_sink: BackedUpSink) -> None:
        num_sinks_current = min(self.sink_size, batch_context_len) // key_cache.shape[-2]
        sink_blocks = decode_meta.block_tables[batch_i, :num_sinks_current]
        sink_key_cache = torch.index_select(key_cache, index=sink_blocks, dim=0)
        backed_up_sink.append(sink_key_cache.clone(), sink_blocks)

    def _rotate_sinks(self, decode_meta: MagicMock, key_cache: torch.Tensor, rotary_emb: MagicMock,
                     backed_up_sink: BackedUpSink):
        """Perform rotation on sinks and put it rotated in the cache."""
        for batch_i, backup, blocks in backed_up_sink:
            self._rotate_sink_positions(backup, blocks, decode_meta, key_cache, rotary_emb, batch_i)

    def _rotate_sink_positions(self, backup: torch.Tensor, blocks: torch.Tensor, decode_meta,
                              key_cache: torch.Tensor, rotary_emb: MagicMock, batch_i: int) -> None:
        # get rotations angles
        num_total_tokens_evicted = self._calculate_evictions(decode_meta, batch_i)
        rotation_positions = (torch.ones(1, self.sink_size).to(key_cache.device) * num_total_tokens_evicted).to(int)

        # rotate
        sink_to_rotate = self._format_key_cache_to_rotation(backup)
        dummy_query = torch.zeros_like(sink_to_rotate).to(key_cache.device)
        _, rotated_sinks = rotary_emb(rotation_positions, dummy_query, sink_to_rotate)

        # Restore rotated sinks into the original position in the cache
        key_cache[blocks[0]] = rotated_sinks.view(rotated_sinks.shape[0],   # fixme: only the first block
                                                  self.num_kv_heads,
                                                  self.head_size//8, 8).permute(1, 2, 0, 3)

    def _format_key_cache_to_rotation(self, x):
        # in: bs,  num_kv_heads, self.head_size/8, 16, 8
        return x.permute(3, 0, 1, 2, 4).reshape(self.sink_size, -1)

    def _calculate_evictions(self, decode_meta: MagicMock, batch_i: int):
        return max(decode_meta.context_lens[batch_i] - self.cache_size, 0)

# TODO:
# add restoring cache from bckp
# add reshpes as special functions
# add parametrization of shapes
# add typing
# pass correct context length


# @pytest.fixture
# def setup_model_evaluator():
#     # Initial settings for the ModelEvaluator class.
#     sink_size = 4
#     sw_size = 4
#     num_kv_heads = 16
#     head_size = 8
#     block_size = 8
#
#     # Create an instance of SinkAttentionRotaryImpl
#     sink_attn_obj = SinkAttentionRotaryImpl(sink_size, sw_size, num_kv_heads, head_size)
#
#     # Prepare mock data for decode_metadata and other inputs
#     key_cache = torch.rand(10, num_kv_heads, block_size, head_size)  # Dummy key_cache tensor
#     decode_meta = MagicMock()
#     attn_metadata = MagicMock()
#     decode_meta.context_lens = torch.LongTensor([8, 9])  # Example context lengths
#     decode_meta.block_tables = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # Simulated block tables
#     attn_metadata.decode_meta = decode_meta
#     def _fake_rotate(x, pos):
#         x.add_(pos[0, :, None].repeat(1, x.shape[-1]))
#         return x
#     rotary_emb = MagicMock(side_effect=lambda positions, dummy_query, sink_key: (_fake_rotate(dummy_query, positions),
#                                                                                  _fake_rotate(sink_key, positions)
#                                                                                  ))
#
#     return sink_attn_obj, key_cache, attn_metadata, rotary_emb


CONTEXT_LEN_1 = list(range(1, 15))
@pytest.fixture
def setup_environment_for_sink(request):
    # Configure a mock environment for testing
    MAX_BLOCK_PER_ONE_EL = 15
    batch_size = 2
    num_blocks = MAX_BLOCK_PER_ONE_EL * batch_size
    block_size = 3
    num_kv_heads = 5
    head_size = 128
    sliding_window = 6
    sink_size = 3

    context_len1 = request.param
    context_len2 = 1 + (context_len1 + 10) % MAX_BLOCK_PER_ONE_EL       # THIS CANNOT BE 0

    hidden_block_size = block_size * num_kv_heads * head_size
    original_shape = (num_kv_heads, block_size, head_size)

    # Creating dummy tensors for key_cache and dummy_query (assuming dtype=torch.float32 for simplicity)
    key_cache = torch.rand(num_blocks, num_kv_heads, head_size//8, block_size, 8)
    decode_meta = MagicMock()  # Simulating the metadata
    decode_meta.block_tables = torch.arange(num_blocks).__reversed__().reshape(batch_size, -1)
    decode_meta.context_lens = torch.LongTensor((context_len1, context_len2))
    attn_metadata = MagicMock()
    attn_metadata.decode_metadata = decode_meta
    sink_attn_obj = SinkAttentionRotaryImpl(sink_size, sliding_window, num_kv_heads, head_size)

    def _fake_rotate(x, pos):
        x.add_(pos[0, :, None].repeat(1, x.shape[-1]))
        return x

    rotary_emb = MagicMock(side_effect=lambda positions, dummy_query, sink_key: (_fake_rotate(dummy_query, positions),
                                                                                 _fake_rotate(sink_key, positions)
                                                                                 ))
    return sink_attn_obj, key_cache, attn_metadata, rotary_emb


@pytest.mark.parametrize("setup_environment_for_sink", CONTEXT_LEN_1, indirect=True)
def test_process_decode_metadata(setup_environment_for_sink):
    sink_attn_obj, key_cache, attn_metadata, rotary_emb = setup_environment_for_sink
    initial_key_cache = key_cache.clone()
    # Run the method under test
    backed_up_sink = sink_attn_obj.process_decode_metadata(attn_metadata, key_cache, rotary_emb)
    # key cache should be updated
    assert torch.max(key_cache - initial_key_cache) > 0
    assert torch.min(key_cache - initial_key_cache) == 0
    assert torch.all(((key_cache - initial_key_cache) % 1.0) == 0)

    # Check if rotary_emb was called, indicating rotation occurred
    sink_attn_obj.restore_cache_from_backup(key_cache, backed_up_sink)
    assert torch.equal(key_cache, initial_key_cache)

    assert rotary_emb.call_count == 1  # Assuming it should be called once in this scenario


def test_process_decode_metadata_and_restore(setup_model_evaluator):
    evaluator, key_cache, attn_metadata, rotary_emb = setup_model_evaluator

    # Run the method under test
    evaluator.process_decode_metadata(attn_metadata, key_cache, rotary_emb)

    # Check if rotary_emb was called, indicating rotation occurred
    # rotary_emb.assert_called()
    assert rotary_emb.call_count == 2  # Assuming it should be called twice in this scenario


def test_prepare_and_restore_sinks(setup_model_evaluator):
    evaluator, key_cache, decode_meta, rotary_emb = setup_model_evaluator

    # Run preparation of sinks
    backups, blocks = evaluator._prepare_sink_rotation(decode_meta, key_cache, )

    # Assert backups and blocks are correctly populated
    assert len(backups) > 0
    assert len(blocks) > 0

    # Simulate a rotation process (mock)
    evaluator._rotate_sinks(decode_meta, key_cache, rotary_emb, backups)

    # Verify that restoration function is called correctly
    assert rotary_emb.call_count > 0


def test_uprotate_sink_single_batch_integration(setup_model_evaluator):
    evaluator, key_cache, decode_meta, rotary_emb = setup_model_evaluator

    # Test the rotation logic
    batch_i, batch_context_len = 0, decode_meta.context_lens[0]
    result = evaluator._uprotate_sink_single_batch(batch_i, batch_context_len, decode_meta, key_cache, key_cache, rotary_emb, evaluator.sink_size, evaluator.cache_size)

    # Assert the rotation produced expected output
    assert result is not None
    backed_up, blocks = result
    assert len(backed_up) == evaluator.sink_size
    assert len(blocks) > 0


# Usage
# evaluator = ModelEvaluator(sink_size=20, cache_size=100, num_kv_heads=16, head_size=64)
# evaluator.process_decode_metadata(attn_metadata, key, key_cache, rotary_emb_function)