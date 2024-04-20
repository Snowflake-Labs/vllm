from xformers.ops.fmha.attn_bias import BlockDiagonalCausalLocalAttentionMask, BlockDiagonalMask
from dataclasses import dataclass
from typing import (
    Optional,
    Tuple,
    Union,
)
import torch


def _materialize_causal_mask_with_sink(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
    *,
    window_size: Optional[int] = None,
    sink_size: Optional[int] = None,
    from_bottomright: bool = False,
) -> torch.Tensor:
    create_as = dtype if dtype is not torch.bfloat16 else torch.float32
    tensor = torch.full(  # type: ignore
        shape,
        dtype=create_as,
        fill_value=1,
        device=device,
    )

    num_queries, num_keys = shape[-2:]
    shift = 0
    if from_bottomright:
        shift = num_keys - num_queries

    mask = torch.tril(tensor, diagonal=shift).to(dtype)  # type: ignore
    if window_size is not None:
        mask = torch.triu(mask, diagonal=shift - window_size + 1)
    if sink_size is not None:
        mask[:, :sink_size] = 1
        mask = torch.tril(mask)
    mask = torch.log(mask)
    return mask.to(dtype)


@dataclass
class BlockDiagonalCausalLocalAttentionAndSinkMask(BlockDiagonalMask):
    """
    (Experimental feature)
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask`.
    This makes the mask "local" and the attention pattern banded.

    Query i only attends to keys in its block and cannot attend keys further than "window_size"
    from it.
    """

    _window_size: int = 0  # forced due to inheritance and default arguments
    _sink_size: int = 0

    def __post_init__(self):
        if self._window_size <= 0:
            raise ValueError(
                f"Expected `window_size > 0`, but window_size={self._window_size}"
            )
        q_seqlen = [
            y - x
            for x, y in zip(
                self.q_seqinfo.seqstart_py[:-1], self.q_seqinfo.seqstart_py[1:]
            )
        ]
        kv_seqlen = [
            y - x
            for x, y in zip(
                self.k_seqinfo.seqstart_py[:-1], self.k_seqinfo.seqstart_py[1:]
            )
        ]
        for q, k in zip(q_seqlen, kv_seqlen):
            if q - self._window_size >= k:
                # Each query only attends to keys no further than window_size back.
                # When q > k + window_size, there will be a query for which the window doesn't reach any key.
                raise RuntimeError(
                    f"No keys are attended in q_seqlen {q} k_seqlen {k} with sliding window {self._window_size}"
                )

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        return _materialize_causal_mask_with_sink(
            shape,
            dtype=dtype,
            device=device,
            window_size=self._window_size,
            sink_size=self._sink_size
        )


# q_len = 40
# k_len = 100
# _materialize_causal_mask_with_sink(
#         (q_len, k_len),
#         # dtype=dtype,
#         # device=device,
#         window_size=20,
#         sink_size=10
#     )
