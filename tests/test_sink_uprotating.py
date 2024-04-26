import torch
import pytest
from unittest.mock import MagicMock


CONTEXT_LEN_1 = list(range(1, 15))
@pytest.fixture
def setup_environment(request):
    # Configure a mock environment for testing
    MAX_BLOCK_PER_ONE_EL = 15
    batch_size = 2
    num_blocks = MAX_BLOCK_PER_ONE_EL * batch_size
    block_size = 5
    num_kv_heads = 3
    head_size = 8
    sliding_window = 4
    sink_size = 2

    context_len1 = request.param
    context_len2 = 1 + (context_len1 + 10) % MAX_BLOCK_PER_ONE_EL       # THIS CANNOT BE 0

    hidden_block_size = block_size * num_kv_heads * head_size
    original_shape = (num_kv_heads, block_size, head_size)

    # Creating dummy tensors for key_cache and dummy_query (assuming dtype=torch.float32 for simplicity)
    key_cache = torch.rand(num_blocks, num_kv_heads, block_size, head_size)
    decode_meta = MagicMock()  # Simulating the metadata
    decode_meta.block_tables = torch.arange(num_blocks).__reversed__().reshape(batch_size, -1)
    decode_meta.context_lens = torch.LongTensor((context_len1, context_len2))

    def _fake_rotate(x, pos):
        x.add_(pos[0, :, None].repeat(1, x.shape[-1]))
        return x
    rotary_emb = MagicMock(side_effect=lambda positions, dummy_query, sink_key: (_fake_rotate(dummy_query, positions),
                                                                                 _fake_rotate(sink_key, positions)
                                                                                 ))

    return {
        'key_cache': key_cache,
        'decode_meta': decode_meta,
        'rotary_emb': rotary_emb,
        'sliding_window': sliding_window,
        'sink_size': sink_size,
        'original_shape': original_shape,
    }

@pytest.mark.parametrize("setup_environment", CONTEXT_LEN_1, indirect=True)
def test_rotation_logic_multibatch(setup_environment):
    """testing up-rotation logic for the sink during multibatch setup"""
    env = setup_environment

    for batch_i, batch_i_cl in enumerate(env['decode_meta'].context_lens):
        num_tokens_evicted_this_pass, rotated_sink_keys, sink_blocks = reference_uprotate(batch_i, batch_i_cl,
                                                                                          env['sink_size'],
                                                                                          env['sliding_window'],
                                                                                          env['decode_meta'],
                                                                                          env['key_cache'],
                                                                                          env['rotary_emb'])

        # Update the keys in the key cache directly after rotation
        # Here we assume that sink_blocks contains the indices to be updated
        for i, block_idx in enumerate(sink_blocks):
            assert torch.allclose(rotated_sink_keys[i].view(
                env['original_shape']) - env['key_cache'][block_idx] - num_tokens_evicted_this_pass,
                                  torch.zeros_like(env['key_cache'][block_idx]),
                           atol=0.001)
            env['key_cache'][block_idx] = rotated_sink_keys[i].view(env['original_shape'])

    # Assertions to check the correctness of rotations and keys
    # assert rotate_positions.numel() == sum(env['decode_meta'].context_lens.tolist())
    env['rotary_emb'].assert_called()


def reference_uprotate(batch_i, batch_i_cl, sink_size, sliding_window, decode_meta, key_cache, rotary_emb):
    num_sinks_current = min(sink_size, batch_i_cl)
    sink_blocks = decode_meta.block_tables[batch_i, :sink_size]
    sink_key_cache = torch.index_select(key_cache, index=sink_blocks, dim=0)
    sink_key_to_roll = sink_key_cache.view(sink_size, -1)
    dummy_query_to_roll = torch.zeros_like(sink_key_to_roll).to(key_cache.device)
    num_tokens_evicted_this_pass = max(batch_i_cl.item() - (sliding_window + sink_size), 0)
    positions_one_bs = torch.ones(1, num_sinks_current).to(key_cache.device) * num_tokens_evicted_this_pass
    _, rotated_sink_keys = rotary_emb(positions_one_bs, dummy_query_to_roll, sink_key_to_roll)
    return num_tokens_evicted_this_pass, rotated_sink_keys, sink_blocks


if __name__ == "__main__":
    pytest.main()
