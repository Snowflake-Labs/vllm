_PAD_SLOT_ID = -1


def test_case(computed_len, prefill_end, sliding_window, prompt_len, sink_size=0,
              block_size=16, block_table = [323386, 323385]):
    # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
    # where start_idx is max(0, prompt_len - sliding_window).
    # For example, if the prompt len is 10, sliding window is 8, and
    # block size is 4, the first two tokens are masked and the slot
    # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
    slot_mapping = []

    start_idx = 0
    if sliding_window is not None:
        assert computed_len == 0, (
            "Prefix caching is currently not supported with "
            "sliding window attention")
        start_idx = max(0, prompt_len - sliding_window)

    for i in range(computed_len, prefill_end):
        if i < start_idx and i >= sink_size:
            slot_mapping.append(_PAD_SLOT_ID)
            continue

        block_number = block_table[i // block_size]
        block_offset = i % block_size
        slot = block_number * block_size + block_offset
        slot_mapping.append(slot)

    return slot_mapping


if __name__ == "__main__":
    slot_mapping = test_case(computed_len=0, prefill_end=26, sliding_window=32, prompt_len=26)
    assert slot_mapping==[5174176, 5174177, 5174178, 5174179, 5174180, 5174181, 5174182, 5174183, 5174184, 5174185, 5174186, 5174187, 5174188, 5174189, 5174190, 5174191, 5174160, 5174161, 5174162, 5174163, 5174164, 5174165, 5174166, 5174167, 5174168, 5174169]

    # sliding window masks out the initial tokens
    slot_mapping = test_case(computed_len=0, prefill_end=26, sliding_window=16, prompt_len=26)
    assert slot_mapping==[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5174186, 5174187, 5174188, 5174189, 5174190, 5174191, 5174160, 5174161, 5174162, 5174163, 5174164, 5174165, 5174166, 5174167, 5174168, 5174169]

    # sliding window masks out the initial tokens, but not sinks
    slot_mapping = test_case(computed_len=0, prefill_end=26, sliding_window=16, prompt_len=26, sink_size=3)
    assert slot_mapping==[5174176, 5174177, 5174178, -1, -1, -1, -1, -1, -1, -1, 5174186, 5174187, 5174188, 5174189, 5174190, 5174191, 5174160, 5174161, 5174162, 5174163, 5174164, 5174165, 5174166, 5174167, 5174168, 5174169]

    slot_mapping = test_case(computed_len=0, prefill_end=10, sliding_window=8, prompt_len=10, block_size=4, block_table=[0, 1, 2])
    assert slot_mapping==[-1, -1, 2, 3, 4, 5, 6, 7, 8, 9]

    slot_mapping = test_case(computed_len=0, prefill_end=10, sliding_window=8, prompt_len=10, sink_size=1, block_size=4, block_table=[0, 1, 2])
    assert slot_mapping==[0, -1, 2, 3, 4, 5, 6, 7, 8, 9]


    slot_mapping = test_case(computed_len=0, prefill_end=10, sliding_window=8, prompt_len=10, sink_size=3, block_size=4, block_table=[0, 1, 2])
    assert slot_mapping==[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # is the same sliding window gonna fit when long contexT?>
    slot_mapping = test_case(computed_len=0, prefill_end=100, sliding_window=8, prompt_len=100, sink_size=3, block_size=4, block_table=[0, 1, 2])
    assert slot_mapping==[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
