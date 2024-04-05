"""Inference-only Yak model."""
from typing import List, Optional, Tuple

import torch
from torch import nn
import os
import torch.nn.functional as F
# TODO(Hao): this assume it will depend on the transformer version which includes Yak
from transformers import YakConfig
from transformers.activations import ACT2FN

import ray

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.fused_moe import fused_topk, fused_experts
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.sequence import SamplerOutput
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.model_executor.layers.quantization.yq import YakQuantizedParameter, YQLinearMethod

KVCache = Tuple[torch.Tensor, torch.Tensor]

use_fused_moe = os.environ.get('USE_FUSE', False)
use_fused_moe = bool(use_fused_moe)
print(f"=== FUSE: {use_fused_moe}===")

m = 0

def pr(x):
    if m == 13:
        print(x)
        exit(0)


class YakMLP(nn.Module):
    def __init__(self, config: YakConfig,
                 layer_id: int,
                 expert_id: int = -1,
                 is_residual_mlp: bool = False,
                 linear_method: Optional[LinearMethodBase] = None,
                 reduce_results: bool = True):
        super(YakMLP, self).__init__()
        self.hidden_size = config.hidden_size
        self.expert_id = expert_id
        self.layer_id = layer_id

        # TODO(Hao): make this tensor-parallel using RowParallelLinear
        self.ffn_dim = config.intermediate_size if not is_residual_mlp else self.hidden_size

        self.w13 = MergedColumnParallelLinear(
            self.hidden_size, [self.ffn_dim] * 2,
            bias=False,
            linear_method=linear_method)
        self.w2 = RowParallelLinear(self.ffn_dim,
                                     self.hidden_size,
                                     bias=False,
                                     reduce_results=reduce_results,
                                     linear_method=linear_method)
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()


    def forward(self, hidden_states):
        gate_up, _ = self.w13(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.w2(hidden_states)
        return hidden_states


class YakMoE(nn.Module):
    """Model-parallel implementation of Yak MoE Layer.

    Note that Yak has *every other* layer to be an MoE (different from Mixtral).
    """

    def __init__(self, config: YakConfig,
                 layer_id: int,
                 tp_size: Optional[int] = None,
                 params_dtype: Optional[torch.dtype] = None,
                 linear_method: Optional[LinearMethodBase] = None,
                 reduce_results: bool = True):
        super(YakMoE, self).__init__()

        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.layer_id = layer_id
        self.top_k = config.num_experts_per_tok
        self.intermediate_size = config.intermediate_size // self.tp_size

        self.is_moe_layer = (layer_id+1) % config.moe_layer_frequency == 0
        self.is_quant = isinstance(linear_method, YQLinearMethod)
        self.reduce_results = reduce_results

        # Some other parameters
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        if not self.is_moe_layer:
            self.mlp = YakMLP(config, layer_id=layer_id, linear_method=linear_method,
                              reduce_results=reduce_results)
        else:
            self.gate = ReplicatedLinear(self.hidden_size,
                                         self.num_experts,
                                         bias=False,
                                         params_dtype=self.params_dtype,
                                         linear_method=linear_method)
            if not use_fused_moe:
                self.experts = nn.ModuleList([
                    YakMLP(
                        config,
                        layer_id=layer_id,
                        expert_id=i,
                        linear_method=linear_method,
                        reduce_results=reduce_results,
                    ) for i in range(self.num_experts)
                ])
            else:
                # Create it on CPU and later quantize it to GPU.
                if self.is_quant:
                    self.ws = YakQuantizedParameter(
                        torch.empty(self.num_experts,
                                    2 * self.intermediate_size,
                                    self.hidden_size,
                                    dtype=self.params_dtype).cpu(),
                        requires_grad=False
                    )
                    self.w2s = YakQuantizedParameter(
                        torch.empty(self.num_experts,
                                    self.hidden_size,
                                    self.intermediate_size,
                                    dtype=self.params_dtype).cpu(),
                        requires_grad=False
                    )
                else:
                    self.ws = nn.Parameter(
                        torch.empty(self.num_experts,
                                    2 * self.intermediate_size,
                                    self.hidden_size,
                                    device="cuda",
                                    dtype=self.params_dtype))
                    self.w2s = nn.Parameter(
                        torch.empty(self.num_experts,
                                    self.hidden_size,
                                    self.intermediate_size,
                                    device="cuda",
                                    dtype=self.params_dtype))
                set_weight_attrs(self.ws, {
                    "weight_loader": self.weight_loader,
                })
                set_weight_attrs(self.w2s, {
                    "weight_loader": self.weight_loader,                
                })

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str, expert_id: int):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        if weight_name.endswith("w1.weight"):
            param_data[expert_id, 0:shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("w3.weight"):
            param_data[expert_id,
                       shard_size:2 * shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("w2.weight"):
            param_data[expert_id, :, :] = loaded_weight[:, shard]

    # Copied from transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock.forward with Mixtral->Yak
    def local_moe(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits, _ = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.top_k > 1:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        # routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_size)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_size)
        return final_hidden_states

    def _selective_dequantize(self, param, topk_ids):
        # select the experts in topk_ids
        tensor = param.view((self.num_experts, -1, param.shape[-1]))
        tensor = tensor.index_select(0, topk_ids.flatten())
        # dequantize the selected experts
        orig_shape = param.quantizer.orig_shape
        param.quantizer.orig_shape = (tensor.shape[0], *orig_shape[1:])
        dequantized = param.quantizer.dequantize(tensor, q_bits=6)
        param.quantizer.orig_shape = orig_shape
        return dequantized

    def local_moe_fused(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits, _ = self.gate(hidden_states)
        do_normalize = True if self.top_k > 1 else False
        topk_weights, topk_ids = fused_topk(hidden_states,
                                            router_logits,
                                            self.top_k,
                                            renormalize=do_normalize)
        # topk_ids: (batch * sequence_length, k)
        if self.is_quant:
            if 2 * topk_ids.numel() <= self.num_experts:
                # If much fewer tokens than experts, use selective dequantize.
                ws_dequantized = self._selective_dequantize(self.ws, topk_ids)
                w2s_dequantized = self._selective_dequantize(self.w2s, topk_ids)
                # We gathered the experts to the tokens so update the mapping.
                topk_ids = torch.arange(
                    0, topk_ids.numel(),
                    device=topk_ids.device,
                ).reshape(topk_ids.shape)
            else:
                ws_dequantized = self.ws.dequantized()
                w2s_dequantized = self.w2s.dequantized()
        final_hidden_states = fused_experts(hidden_states,
                                            ws_dequantized if self.is_quant else self.ws,
                                            w2s_dequantized if self.is_quant else self.w2s,
                                            topk_weights,
                                            topk_ids,
                                            inplace=True)
        # print(f"Exit the kernel, dtype: {self.ws.dtype}, {self.w2s.dtype}")
        if self.reduce_results and self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(batch_size, sequence_length,
                                        hidden_size)


    def forward(self, hidden_states: torch.Tensor):
        if self.is_moe_layer:
            if use_fused_moe:
                final_hidden_states = self.local_moe_fused(hidden_states)
            else:
                final_hidden_states = self.local_moe(hidden_states)
        else:
            final_hidden_states = self.mlp(hidden_states)
        return final_hidden_states


class YakAttention(nn.Module):
    def __init__(self, 
                 config: YakConfig, 
                 layer_idx: Optional[int] = None,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim


        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta        
        self.scaling = self.head_dim**-0.5

        # self.q_proj = nn.Linear(self.hidden_size, self.num_heads // config.tp_size * self.head_dim, bias=False)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads // config.tp_size * self.head_dim, bias=False)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads // config.tp_size * self.head_dim, bias=False)
        # self.o_proj = nn.Linear(self.num_heads // config.tp_size * self.head_dim, self.hidden_size, bias=False)

        # self.rotary_emb = YakRotaryEmbedding(
        #     self.head_dim,
        #     max_position_embeddings=self.max_position_embeddings,
        #     base=self.rope_theta,
        # )

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            linear_method=linear_method,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=int(self.rope_theta),
            is_neox_style=True,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output

class YakDecoderLayer(nn.Module):
    def __init__(
        self,
        config: YakConfig,
        layer_idx: int,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        is_moe_layer = (layer_idx+1) % config.moe_layer_frequency == 0
        self.use_residual = config.use_residual and is_moe_layer
        self.self_attn = YakAttention(config, layer_idx, linear_method=linear_method)
        self.block_sparse_moe = YakMoE(config, layer_id=layer_idx, linear_method=linear_method,
                                       reduce_results=(not self.use_residual))

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.use_residual:
            self.residual_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.residual_mlp = YakMLP(config, layer_id=layer_idx, is_residual_mlp=True,
                                       reduce_results=False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        residual_input = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata
        )
        hidden_states = residual_input + hidden_states

        residual_attn = hidden_states
        if self.use_residual:
            hidden_states = self.residual_layernorm(hidden_states)
            hidden_states = self.residual_mlp(hidden_states)
            residual_mlp = hidden_states
            hidden_states = self.post_attention_layernorm(residual_input)
            hidden_states = self.block_sparse_moe(hidden_states)
            hidden_states = residual_mlp + hidden_states
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
            hidden_states = residual_attn + hidden_states
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.block_sparse_moe(hidden_states)
            hidden_states = residual_attn + hidden_states
        return hidden_states


class YakModel(nn.Module):
    def __init__(
        self,
        config: YakConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        
        # self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=self.vocab_size
        )
        self.layers = nn.ModuleList([
            YakDecoderLayer(config, layer_idx, linear_method=linear_method)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self._attn_implementation = config._attn_implementation
        #TODO(Hao): vllm has its own impl or RMSNorm, should we consider?
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        global m
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(positions, hidden_states,
                                  kv_caches[i], input_metadata)
            m = m + 1
        hidden_states = self.norm(hidden_states)
        return hidden_states


class YakForCausalLM(nn.Module):
    def __init__(
        self, 
        config: YakConfig,
        linear_method: Optional[LinearMethodBase] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.model = YakModel(config, linear_method)
        self.config = config
        self.linear_method = linear_method
        self.vocab_size = config.vocab_size
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head = ParallelLMHead(
            self.vocab_size,
            config.hidden_size,
        )
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.unpadded_vocab_size = config.vocab_size
        # # Initialize weights and apply final processing
        # self.post_init()
        self.sampler = Sampler(self.unpadded_vocab_size, config.vocab_size)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[KVCache],
                input_metadata: InputMetadata,
            ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches, input_metadata)
        return hidden_states

    def sample(
        self,
        hidden_states: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states, sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        mlp_params_mapping = []
        expert_params_mapping = []
        num_layers = self.config.num_hidden_layers

        if use_fused_moe:
            for i in range(num_layers):
                for weight_name in ["w1", "w3"]:
                    mapping = (f"layers.{i}.residual_mlp.w13.weight", 
                               f"layers.{i}.residual_mlp.{weight_name}.weight",
                               0 if weight_name == "w1" else 1)
                    mlp_params_mapping.append(mapping)
                if i % 2 == 0:
                    # MLP layers
                    for weight_name in ["w1", "w3"]:
                        mapping = (f"layers.{i}.block_sparse_moe.mlp.w13.weight", 
                                    f"layers.{i}.block_sparse_moe.mlp.{weight_name}.weight",
                                    0 if weight_name == "w1" else 1)
                        mlp_params_mapping.append(mapping)
                else:
                    # MoE layers
                    for expert_id in range(self.config.num_local_experts):
                        for weight_name in ["w1", "w2", "w3"]:
                            if weight_name in ["w1", "w3"]:
                                mapping = ("ws", f"experts.{expert_id}.{weight_name}.weight", expert_id)
                            else:
                                mapping = ("w2s", f"experts.{expert_id}.{weight_name}.weight", expert_id)
                            expert_params_mapping.append(mapping)
        else:
            mlp_params_mapping = [
                ("w13", "w1", 0),
                ("w13", "w3", 1),
            ]

        params_dict = dict(self.named_parameters())
        loaded_iterators = hf_model_weights_iterator(
            model_name_or_path,
            cache_dir,
            load_format,
            revision,
            fall_back_to_pt=False)
        
        def fused_load():
            for name, loaded_weight in loaded_iterators:
                original_name = name
                for (param_name, weight_name, shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue

                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    print(f"Loaded weight {original_name} with shape {loaded_weight.shape} into module {name} ({param.shape}) as shard {shard_id}")
                    break
                else:
                    for param_name, weight_name, shard_id in mlp_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id)
                        print(f"Loaded weight {original_name} ({loaded_weight.shape}) into module {name} ({param.shape}) as shard {shard_id}")
                        break
                    else:        
                        for param_name, weight_name, shard_id in expert_params_mapping:
                            if weight_name not in name:
                                continue
                            name = name.replace(weight_name, param_name)
                            param = params_dict[name]
                            weight_loader = param.weight_loader
                            weight_loader(param, loaded_weight, weight_name, expert_id=shard_id)
                            print(f"Loaded weight {original_name} ({loaded_weight.shape}) into module {name} ({param.shape}) as shard {shard_id}")
                            break
                        else:
                            if name.endswith(".bias") and name not in params_dict:
                                continue
                            param = params_dict[name]
                            weight_loader = getattr(param, "weight_loader",
                                                    default_weight_loader)
                            weight_loader(param, loaded_weight)
                            print(f"Loaded weight {original_name} ({loaded_weight.shape}) into module {name} ({param.shape}) as shard 0")

        def unfused_load():
            for name, loaded_weight in loaded_iterators:
                print(f"Load weight {name} with shape {loaded_weight.shape}.")
                for (param_name, weight_name, shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    for param_name, weight_name, shard_id in mlp_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id)
                        break
                    else:
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)

        if use_fused_moe:
            fused_load()
        else:
            unfused_load()

        # For yak, we run a post quantization, because the weights are saved in 16 bits.
        # Iterate in order of largest to smallest params to reduce fragmentation issues.
        for name, param in sorted(self.named_parameters(), key=lambda v: -v[1].numel()):
            if hasattr(param, "is_yak") and param.is_yak == True:
                # do quantization after loading and moe to GPU
                assert param.device.type != "cuda"
                param.data = param.cuda()
                print(f"Quantize weight {name} with dtype {param.data.dtype} and shape {param.data.shape}.")
