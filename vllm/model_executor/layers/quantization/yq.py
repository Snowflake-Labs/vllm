from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn 

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
        q_range = 480.0,
        group_size: int = 512,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.rounding = rounding
        self.mantissa_bits = mantissa_bits
        self.q_range = q_range
        self.valid_types = [torch.bfloat16, torch.float16]

        if self.weight_bits not in [6, 8]:
            raise ValueError(
                "Currently, only 6-bit or 8-bit weight quantization are "
                f"supported for Yak quantizaiton, but got {self.weight_bits} bits."
            )

    def __repr__(self) -> str:
        return (f"YQConfig(weight_bits={self.weight_bits}), "
                f"group_size={self.group_size}, "
                f"rounding={self.rounding}, "
                f"mantissa_bits={self.mantissa_bits}, "
                f"")
    @classmethod
    def get_name(cls) -> str:
        return "yq"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "YQConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(weight_bits=weight_bits, group_size=group_size)

    def get_linear_method(self) -> "YQLinearMethod":
        return YQLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json", 
            "quantize_config.json",
        ]


class YQLinearMethod(LinearMethodBase):
    """Linear method for Yak.

    Args:
        quant_config: the Yak quantization config.
    """

    def __init__(self, quant_config: YQConfig):
        print(f"Inside YQLinearMethod init: {quant_config}")
        self.quant_config = quant_config
        self.weight = None
        self.q_weight = None
        # create the quantizer
        from deepspeed.ops.fp_quantizer import FP_Quantize
        self.quantizer = FP_Quantize(
            group_size=self.quant_config.group_size,
        )

    def create_weights(self,
                       input_size_per_partition: int,
                       output_size_per_partition: int,
                       input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, Any]:
        group_size = self.quant_config.group_size
        orig_numel = input_size_per_partition * output_size_per_partition
        num_groups = orig_numel // group_size
        weight = YakQuantizedParameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype
            ).cpu(),
            requires_grad=False,
            quantization=self.quant_config,
        )
        # scales = Parameter(
        #     torch.empty(
        #         num_groups,
        #         4,
        #         dtype=torch.int8
        #     ),
        #     requires_grad=False,
        # )
        set_weight_attrs(
            weight, {
                "input_dim": 1,
                "output_dim": 0,
        })
        return {
            "weight": weight,
        }

    def apply_weights(self,
                      weights: Dict[str, Any],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # print(f"Running Yak Linear Method for dtype {weights['weight'].dtype}")
        weight = weights["weight"]
        return F.linear(x, weight.dequantized(), bias)

    def dequantize(self, weight):
        assert weight.data.dtype == torch.int8 and weight.data.device.type == "cuda"
        with torch.cuda.stream(torch.cuda.current_stream(weight.data.device)):
            return self.quantizer.dequantize(weight.data, self.scales)

    def quantize(self, weight, return_meta_tensor=True):
        # if weight.data.device.type == "cuda" and weight.data.dtype != torch.int8:
        with torch.cuda.stream(torch.cuda.current_stream(weight.data.device)):
            return self.quantizer.quantize(weight.data, return_meta_tensor=return_meta_tensor)

# class YQLinearMethod(LinearMethodBase):
#     """Linear method for Yak.

#     Args:
#         quant_config: the Yak quantization config.
#     """

#     def __init__(self, quant_config: YQConfig):
#         self.quant_config = quant_config
#         self.weight = None
#         self.scales = None
#         # create the quantizer
#         from deepspeed.ops.fp_quantizer import FP_Quantize
#         self.quantizer = FP_Quantize(
#             q_bits=self.quant_config.weight_bits,
#             rounding=self.quant_config.rounding,
#             mantisa_bits=self.quant_config.mantissa_bits,
#             group_size=self.quant_config.group_size,
#         )

#     def create_weights(self,
#                        input_size_per_partition: int,
#                        output_size_per_partition: int,
#                        input_size: int,
#                        output_size: int,
#                        params_dtype: torch.dtype) -> Dict[str, Any]:
#         group_size = self.quant_config.group_size
#         orig_numel = input_size_per_partition * output_size_per_partition
#         num_groups = orig_numel // group_size
#         weight = Parameter(
#             torch.empty(
#                 output_size_per_partition,
#                 input_size_per_partition,
#                 dtype=torch.int8
#             ),
#             requires_grad=False,
#         )
#         scales = Parameter(
#             torch.empty(
#                 num_groups,
#                 4,
#                 dtype=torch.int8
#             ),
#             requires_grad=False,
#         )
#         set_weight_attrs(
#             weight, {
#                 "input_dim": 1,
#                 "output_dim": 0,
#                 "is_yak": True,
#                 "scales": scales,
#             })
#         return {
#             "weight": weight,
#         }

#     def apply_weights(self,
#                       weights: Dict[str, Any],
#                       x: torch.Tensor,
#                       bias: Optional[torch.Tensor] = None) -> torch.Tensor:
#         print(f"Running Yak Linear Method for dtype {weights['weight'].dtype}")
#         weight = weights["weight"]
#         return F.linear(x, self.dequantize(weight), bias)

#     def dequantize(self, weight):
#         assert weight.data.dtype == torch.int8 and weight.data.device.type == "cuda"
#         with torch.cuda.stream(torch.cuda.current_stream(weight.data.device)):
#             return self.quantizer.dequantize(weight.data, self.scales)

#     def quantize(self, weight, return_meta_tensor=True):
#         # if weight.data.device.type == "cuda" and weight.data.dtype != torch.int8:
#         with torch.cuda.stream(torch.cuda.current_stream(weight.data.device)):
#             return self.quantizer.quantize(weight.data, return_meta_tensor=return_meta_tensor)


    

# class YQLinearMethod2(LinearMethodBase):
#     """Linear method for Yak.

#     Args:
#         quant_config: the Yak quantization config.
#     """

#     def __init__(self, quant_config: YQConfig):
#         self.quant_config = quant_config
#         self.weight = None
#         self.quantizer = None

#     def create_weights(self,
#                        input_size_per_partition: int,
#                        output_size_per_partition: int,
#                        input_size: int,
#                        output_size: int,
#                        params_dtype: torch.dtype) -> Dict[str, Any]:
#         group_size = self.quant_config.group_size
#         orig_numel = input_size_per_partition * output_size_per_partition
#         num_groups = orig_numel // self.quant_config.group_size
#         self.weight = torch.empty(
#             output_size_per_partition,
#             input_size_per_partition,
#             dtype=torch.bfloat16,
#         ).cpu()
#         self.qweight = torch.empty(
#             num_groups,
#             group_size + 4,
#             # output_size_per_partition,
#             dtype=torch.int8)
#         weight = YakQuantizedParameter(self.weight, self.qweight, quantization=self.quant_config)
#         set_weight_attrs(
#             weight, {
#                 "output_dim": 0,
#             })
#         return {
#             "weight": weight
#         }

#     def apply_weights(self,
#                       weights: Dict[str, Any],
#                       x: torch.Tensor,
#                       bias: Optional[torch.Tensor] = None) -> torch.Tensor:

#         if self.quantizer is not None:
#             from deepspeed.ops.fp_quantizer import FP_Quantize
#             quantization = self.quant_config
#             self.quantizer = FP_Quantize(
#                 q_bits=quantization.weight_bits,
#                 rounding=quantization.rounding,
#                 mantisa_bits=quantization.mantissa_bits,
#                 group_size=quantization.group_size,
#             )
#         weight = weights["weight"]
#         print("111111")
#         return F.linear(x, weight.dequantized(), bias)

#     def _dequantize(self):
#         if self.data.device.type == "cuda" and self.data.dtype == torch.int8:
#             with torch.cuda.stream(torch.cuda.current_stream(self.data.device)):
#                 return self.quantizer.dequantize(self.data)
#         return self.data



class YakQuantizedParameter(nn.Parameter):
    """
    Yak parameter class that implements weight quantization via deepspeed. Weights
    are stored in quantized form on GPUs, and can be dequantized on-the-fly when
    needed by the model. The weights are actually quantized during any `.to(device)`
    if `device` is a cuda device.
    """
    def __new__(
        cls,
        data,
        requires_grad: bool = False,  # quantized weights should be frozen by default
        quantization: YQConfig = None,
        quantizer=None,  # HF expects this argument.
    ):
        if quantization is None:
            quantization = YQConfig()
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.quant_config = quantization
        self.data = data
        from deepspeed.ops.fp_quantizer import FP_Quantize
        if quantizer is not None:
            self.quantizer = quantizer
        else:
            self.quantizer = FP_Quantize(
                group_size=quantization.group_size,
            )
        self._ensure_quantized(self)
        return self

    def _ensure_quantized(self, tensor: torch.Tensor):
        # If the tensor is on a cuda device and is not quantized, then quantize it in-place.
        if tensor.device.type == "cuda" and tensor.dtype != torch.int8:
            with torch.cuda.stream(torch.cuda.current_stream(tensor.device)):
                tensor.data = self.quantizer.quantize(tensor.data, q_bits=self.quant_config.weight_bits)
            assert tensor.dtype == torch.int8

    def dequantized(self) -> torch.Tensor:
        """
        Return a tensor containing the dequantized weights of this parameter.
        """
        if self.data.device.type == "cuda" and self.data.dtype == torch.int8:
            with torch.cuda.stream(torch.cuda.current_stream(self.data.device)):
                return self.quantizer.dequantize(self.data, q_bits=self.quant_config.weight_bits)
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

    @property
    def is_yak(self):
        return True

    def cuda_parameter(self, device=None, non_blocking=False):
        a = self.to(device="cuda" if device is None else device, non_blocking=non_blocking)
        return torch.Tensor._make_subclass(YakQuantizedParameter, a, self.requires_grad)
