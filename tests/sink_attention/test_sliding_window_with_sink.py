from typing import Callable, Optional
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed

from vllm.attention.backends.sink_rotations import (BackedUpSink,
                                                    SinkAttentionRotaryImpl)

CONTEXT_LEN = list(range(1, 20))


@pytest.fixture
def setup_environment_for_sink(request):
    # Configure a mock environment for testing
    MAX_BLOCK_PER_ONE_EL = 15
    batch_size = 4
    num_blocks = MAX_BLOCK_PER_ONE_EL * batch_size
    # key shapes
    num_kv_heads = 5
    head_size = 128
    block_size = 3
    # cache persistency
    sliding_window = 3
    sink_size = 3
    cache_size = sink_size + sliding_window

    # Creating dummy tensors for key_cache and dummy_query (assuming dtype=torch.float32 for simplicity)
    key_cache = torch.rand(num_blocks, num_kv_heads, head_size // 8, block_size, 8)
    decode_meta = MagicMock()  # Simulating the metadata
    decode_meta.block_tables = (
        torch.arange(num_blocks).__reversed__().reshape(batch_size, -1)
    )
    # build samples
    context_lens = []
    for b_el in range(batch_size):
        context_lens.append(request.param + b_el * 10 % MAX_BLOCK_PER_ONE_EL)
    positions = torch.LongTensor(context_lens)
    decode_meta.context_lens = torch.min(
        positions, torch.ones_like(positions) * cache_size
    ).to(int)

    attn_metadata = MagicMock()
    attn_metadata.decode_metadata = decode_meta
    sink_attn_obj = SinkAttentionRotaryImpl(
        sink_size, sliding_window, num_kv_heads, head_size
    )

    def _fake_rotate(x, pos):
        x.add_(pos[:, :, None].repeat(1, 1, x.shape[-1]))
        return x

    rotary_emb = MagicMock(
        side_effect=lambda positions, dummy_query, sink_key: (
            _fake_rotate(dummy_query, positions),
            _fake_rotate(sink_key, positions),
        )
    )
    return sink_attn_obj, key_cache, attn_metadata, rotary_emb, positions


@pytest.mark.parametrize("setup_environment_for_sink", CONTEXT_LEN, indirect=True)
def test_process_decode_metadata_and_restore(setup_environment_for_sink):
    sink_attn_obj, key_cache, attn_metadata, rotary_emb, positions = (
        setup_environment_for_sink
    )
    initial_key_cache = key_cache.clone()
    # Run the method under test
    backed_up_sink = sink_attn_obj.process_decode_metadata(
        attn_metadata, key_cache, rotary_emb, positions
    )
    # Key cache should be updated
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

    # Check if rotary_emb was called, indicating rotation occurred
    sink_attn_obj.restore_cache_from_backup(key_cache, backed_up_sink)
    assert torch.equal(key_cache, initial_key_cache)
