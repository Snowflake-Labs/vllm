from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from vllm._C import ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

class YQConfig(QuantizationConfig):
    """Config for YQ."""

    def __init__(
        self,
        weight_bits: int = 8,
        rounding: str = "nearest",
        mantissa_bits: int = 3,
        group_size: int = 512
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.rounding = rounding
        self.mantissa_bits = mantissa_bits

        if self.weight_bits != 8:
            raise ValueError(
                "Currently, only 8-bit weight quantization is supported for "
                f"Yak quantizaiton, but got {self.weight_bits} bits."
            )
    def __repr__(self) -> str:
        return (f"YQConfig(weight_bits={self.weight_bits}), "
                f"group_size={self.group_size}, "
                f"rounding={self.rounding}, "
                f"mantissa_bits={self.mantissa_bits}")
    @classmethod
    def get_name(cls) -> str:
        return "yq"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "YQConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(weight_bis=weight_bits, group_size=group_size)

    def get_linear_method(self) -> "YQLinearMethod":
        return YQLinearMethod(self)


class YQLinearMethod(LinearMethodBase):
    """Linear method for Yak.

    Args:
        quant_config: the Yak quantization config.
    """

    def __init__(self, quant_config: yq):
        self.quant_config = quant_config
        self.weight = None

    def create_weights(self, input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, Any]:
        self.weight = torch.empty(
            input_size_per_partition,
            output_size_per_partition,
            dtype=torch.int8,
        )
        qweight = YakQuantizedParameter(self.weight, quantization=self.quant_config)
        return {
            "qweight": qweight
        }
        # TODO(Hao): do the set_weight_attr things

    def apply_weights(self,
                      weights: Dict[str, Any],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweights = weights["qweight"]
        return F.linear(x, qweight.dequantized(), bias)


class YakQuantizedParameter(nn.Parameter):
    """
    Yak parameter class that implements weight quantization via deepspeed. Weights
    are stored in quantized form on GPUs, and can be dequantized on-the-fly when
    needed by the model. The weights are actually quantized during any `.to(device)`
    if `device` is a cuda device.
    """

    def __new__(
        cls,
        data: Optional[torch.Tensor] = None,
        requires_grad: bool = False,  # quantized weights should be frozen by default
        quantization: YakQuantizationConfig = None,
        quantizer=None,  # HF expects this argument.
    ):
        if data is None:
            data = torch.empty(0)
        if quantization is None:
            quantization = YakQuantizationConfig()
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        from deepspeed.tops import FP_Quantize
        if quantizer is not None:
            self.quantizer = quantizer
        else:
            self.quantizer = FP_Quantize(
                q_bits=quantization.q_bits,
                rounding=quantization.rounding,
                mantisa_bits=quantization.mantissa_bits,
                group_size=quantization.group_size,
            )
        self._ensure_quantized(self)
        return self

    def _ensure_quantized(self, tensor: torch.Tensor):
        # If the tensor is on a cuda device and is not quantized, then quantize it in-place.
        if tensor.device.type == "cuda" and tensor.dtype != torch.int8:
            with torch.cuda.stream(torch.cuda.current_stream(tensor.device)):
                tensor.data = self.quantizer.quantize(tensor.data)
            assert tensor.dtype == torch.int8

    def dequantized(self) -> torch.Tensor:
        """
        Return a tensor containing the dequantized weights of this parameter.
        """
        if self.data.device.type == "cuda" and self.data.dtype == torch.int8:
            with torch.cuda.stream(torch.cuda.current_stream(self.data.device)):
                return self.quantizer.dequantize(self.data)
        return self.data

    def __getstate__(self):
        state = self.__dict__
        state["data"] = self.data
        state["requires_grad"] = self.requires_grad
        return state

    def __setstate__(self, state):
        self.quantizer = state["quantizer"]
        self.data = state["data"]
        self.requires_grad = state["requires_grad"]

    def __deepcopy__(self, memo):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        new_instance.quantizer = copy.deepcopy(state["quantizer"])
        new_instance.data = copy.deepcopy(state["data"])
        return new_instance

    def __copy__(self):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        return new_instance

    def cuda(self, device=None, non_blocking=False):
        return self.to(device="cuda" if device is None else device, non_blocking=non_blocking)

    def to(self, *args, **kwargs):
        """
        Move the parameter to the given device. Then, if the device is a cuda device,
        quantize it.
        """
        tensor = super().to(*args, **kwargs)
        self._ensure_quantized(tensor)
        return tensor
