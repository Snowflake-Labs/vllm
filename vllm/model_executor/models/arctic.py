"""Inference-only Snowflake Arctic model."""
from typing import Iterable, List, Optional, Tuple
import time

import torch
from torch import nn
import torch.distributed

from vllm.attention import Attention, AttentionMetadata
from vllm.distributed import (get_pipeline_model_parallel_rank,
                              get_pipeline_model_parallel_next_rank,
                              get_pipeline_model_parallel_prev_rank,
                              get_pipeline_model_parallel_world_size,
                              get_pp_indices, get_tensor_model_parallel_rank,
                              get_pp_indices_arctic,
                              get_tensor_model_parallel_world_size,
                              is_pipeline_model_parallel_first_rank,
                              is_pipeline_model_parallel_last_rank,
                              recv_prev_rank, send_next_rank,
                              tensor_model_parallel_all_reduce)
from vllm.config import CacheConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_experts, fused_topk
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.deepspeedfp import (
    DeepSpeedFPConfig, DeepSpeedFPParameter)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs.arctic import ArcticConfig

import logging
import gc
logger = init_logger(__name__)
logger.setLevel(logging.DEBUG)

import time
def _sleep_infinity():
    while(1):
        time.sleep(1)

GB = 1024 * 1024 * 1024


def report_memory(label):
    free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
    memory_allocated = torch.cuda.memory_allocated()
    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(f"Rank {torch.distributed.get_rank()}, {label}, free gpu memory: {free_gpu_memory / GB}, total gpu memory: {total_gpu_memory / GB}, "
          f"memory allocated: {memory_allocated / GB} , max_memory_allocated: {max_memory_allocated / GB}...")


class ArcticMLP(nn.Module):

    def __init__(self,
                 config: ArcticConfig,
                 layer_id: int,
                 expert_id: int = -1,
                 is_residual_mlp: bool = False,
                 quant_config: Optional[QuantizationConfig] = None,
                 reduce_results: bool = True):
        super(ArcticMLP, self).__init__()
        self.hidden_size = config.hidden_size
        self.expert_id = expert_id
        self.layer_id = layer_id

        self.ffn_dim = config.intermediate_size if not is_residual_mlp \
            else self.hidden_size

        self.w13 = MergedColumnParallelLinear(self.hidden_size,
                                              [self.ffn_dim] * 2,
                                              bias=False,
                                              quant_config=quant_config)
        self.w2 = RowParallelLinear(self.ffn_dim,
                                    self.hidden_size,
                                    bias=False,
                                    reduce_results=reduce_results,
                                    quant_config=quant_config)
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states):
        gate_up, _ = self.w13(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.w2(hidden_states)
        return hidden_states


class ArcticMoE(nn.Module):
    """
    Model-parallel implementation of Arctic MoE Layer.
    """

    def __init__(self,
                 config: ArcticConfig,
                 layer_id: int,
                 tp_size: Optional[int] = None,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 reduce_results: bool = True):
        super(ArcticMoE, self).__init__()

        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.layer_id = layer_id
        self.top_k = config.num_experts_per_tok
        self.intermediate_size = config.intermediate_size // self.tp_size
        self.enable_dequantization_fusion = config.enable_dequantization_fusion
        self.is_moe_layer = (layer_id + 1) % config.moe_layer_frequency == 0
        self.is_quant = isinstance(quant_config, DeepSpeedFPConfig)
        self.reduce_results = reduce_results
        # Some other parameters
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        if not self.is_moe_layer:
            self.mlp = ArcticMLP(config,
                                 layer_id=layer_id,
                                 quant_config=quant_config,
                                 reduce_results=reduce_results)
        else:
            self.gate = ReplicatedLinear(self.hidden_size,
                                         self.num_experts,
                                         bias=False,
                                         params_dtype=self.params_dtype,
                                         quant_config=quant_config)
            if self.is_quant:
                self.ws = DeepSpeedFPParameter(
                    torch.Size((self.num_experts, 2 * self.intermediate_size,
                                self.hidden_size)),
                    params_dtype=params_dtype,
                    quant_config=quant_config,
                )
                self.w2s = DeepSpeedFPParameter(
                    torch.Size((self.num_experts, self.hidden_size,
                                self.intermediate_size)),
                    params_dtype=params_dtype,
                    quant_config=quant_config,
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
        self.load_completion_w1_w3 = 0
        self.load_completion_w2 = 0

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str, expert_id: int):
        tp_rank = get_tensor_model_parallel_rank()
        if not hasattr(param, 'shadow_data'):
            param.shadow_data = [None] * self.num_experts
        shard_size = self.intermediate_size
        if param.shadow_data[expert_id] is None:
            param.shadow_data[expert_id] = torch.zeros(
                (shard_size * 2 if (weight_name.endswith("w1.weight") or weight_name.endswith("w3.weight")) else loaded_weight.shape[0],
                loaded_weight.shape[1] if (weight_name.endswith("w1.weight") or weight_name.endswith("w3.weight")) else shard_size), 
                dtype=loaded_weight.dtype, 
                device=loaded_weight.device)
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        if weight_name.endswith("w1.weight"):
            param.shadow_data[expert_id][0:shard_size, :] = loaded_weight[shard, :]
            self.load_completion_w1_w3 += 1
        if weight_name.endswith("w3.weight"):
            param.shadow_data[expert_id][shard_size:, :] = loaded_weight[shard, :]
            self.load_completion_w1_w3 += 1
        if weight_name.endswith("w2.weight"):
            param.shadow_data[expert_id][:, :shard_size] = loaded_weight[:, shard]
            self.load_completion_w2 += 1
        if (self.load_completion_w2 == self.num_experts or self.load_completion_w1_w3 == self.num_experts * 2):
            new_data = torch.stack(param.shadow_data).to(param.data.device)

            len_sd = len(param.shadow_data)
            for _ in range(len_sd):
                sd = param.shadow_data.pop()
                del sd
            del param.shadow_data
            
            if self.load_completion_w2 == self.num_experts:
                self.load_completion_w2 = 0
            if self.load_completion_w1_w3 == self.num_experts * 2:
                self.load_completion_w1_w3 = 0 
            
            if self.is_quant:
                param.ds_quantize_(new_data)
                del new_data
                new_data = None
                gc.collect()
                torch.cuda.empty_cache()
            else:
                param.data.copy_(new_data)

    def local_moe_fused(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        do_normalize = self.top_k > 1
        topk_weights, topk_ids = fused_topk(hidden_states,
                                            router_logits,
                                            self.top_k,
                                            renormalize=do_normalize)
        # topk_ids: (num_tokens, k)
        if self.is_quant and (not self.enable_dequantization_fusion):
            if 2 * num_tokens <= self.num_experts:
                # If much fewer tokens than experts, use selective dequantize.
                ws_dequantized = self.ws.ds_selective_dequantize(
                    topk_ids.flatten())
                w2s_dequantized = self.w2s.ds_selective_dequantize(
                    topk_ids.flatten())
                # We gathered the experts to the tokens so update the mapping.
                topk_ids = torch.arange(
                    0,
                    topk_ids.numel(),
                    device=topk_ids.device,
                ).reshape(topk_ids.shape)
            else:
                ws_dequantized = self.ws.ds_dequantize()
                w2s_dequantized = self.w2s.ds_dequantize()
            # report_memory(f"PP rank: {get_pipeline_model_parallel_rank()} after local quant")
        final_hidden_states = fused_experts(
            hidden_states,
            ws_dequantized if (self.is_quant and not self.enable_dequantization_fusion) else self.ws,
            w2s_dequantized if (self.is_quant and not self.enable_dequantization_fusion) else self.w2s,
            topk_weights,
            topk_ids,
            inplace=True,
            w1_scale=self.ws.quantization_scales() if self.is_quant else None,
            w2_scale=self.w2s.quantization_scales() if self.is_quant else None,
            quantization_group_size=self.ws.fp_quantizer.group_size if self.is_quant else 256,
            quantization_group_size2=self.w2s.fp_quantizer.group_size if self.is_quant else 256)
        if self.reduce_results and self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        # report_memory(f"PP rank: {get_pipeline_model_parallel_rank()} before local fused_moe")
        if self.is_moe_layer:
            final_hidden_states = self.local_moe_fused(hidden_states)
        else:
            final_hidden_states = self.mlp(hidden_states)
        # report_memory(f"PP rank: {get_pipeline_model_parallel_rank()} after local fused_moe")
        return final_hidden_states


class ArcticAttention(nn.Module):

    def __init__(
        self,
        config: ArcticConfig,
        layer_idx: Optional[int] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
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

        self.qkv_proj = QKVParallelLinear(self.hidden_size,
                                          self.head_dim,
                                          self.total_num_heads,
                                          self.total_num_kv_heads,
                                          bias=False,
                                          quant_config=quant_config)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=int(self.rope_theta),
            is_neox_style=True,
        )

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class ArcticDecoderLayer(nn.Module):

    def __init__(
        self,
        config: ArcticConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        is_moe_layer = (layer_idx + 1) % config.moe_layer_frequency == 0
        self.use_residual = config.use_residual and is_moe_layer
        self.self_attn = ArcticAttention(config,
                                         layer_idx,
                                         cache_config,
                                         quant_config=quant_config)
        self.block_sparse_moe = ArcticMoE(
            config,
            layer_id=layer_idx,
            quant_config=quant_config,
            reduce_results=(not self.use_residual))

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

        if self.use_residual:
            self.residual_layernorm = RMSNorm(config.hidden_size,
                                              eps=config.rms_norm_eps)
            self.residual_mlp = ArcticMLP(config,
                                          layer_id=layer_idx,
                                          is_residual_mlp=True,
                                          reduce_results=False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        residual_input = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
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


class ArcticModel(nn.Module):

    def __init__(
        self,
        config: ArcticConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=self.vocab_size)

        # Construct stages
        self.start_layer, self.end_layer = get_pp_indices_arctic(
            config.num_hidden_layers, get_pipeline_model_parallel_rank(),
            get_pipeline_model_parallel_world_size())
        print(f"PP rank {get_pipeline_model_parallel_rank()}, start layer: {self.start_layer}, end layer: {self.end_layer}")
        layers_1 = [nn.Identity() for _ in range(self.start_layer)]
        layers_2 = [ArcticDecoderLayer(config, layer_idx, cache_config=cache_config, quant_config=quant_config) for layer_idx in range(self.start_layer, self.end_layer)]
        layers_3 = [nn.Identity() for _ in range(self.end_layer, config.num_hidden_layers)]

        self.layers = nn.ModuleList(layers_1 + layers_2 + layers_3)
        self._attn_implementation = config._attn_implementation
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if is_pipeline_model_parallel_first_rank():
            hidden_states = self.embed_tokens(input_ids)
        else:
            sizes = list(input_ids.size()) + [self.config.hidden_size]
            hidden_states = recv_prev_rank(1, sizes, self.embed_tokens.weight.dtype,
                                            self.embed_tokens.weight.device)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            # print(f"PP rank {get_pipeline_model_parallel_rank()} TP rank {get_tensor_model_parallel_rank()}: "
            #       f"recv done from {get_pipeline_model_parallel_prev_rank()}, "
            #       f"shape: {hidden_states.shape}, device: {hidden_states.device}, for request with {input_ids.size()} tokens...")

        
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            # print(f"PP rank {get_pipeline_model_parallel_rank()}: layer {i} to start, kv size: {len(kv_caches)}, "
                        #  f"postions: {positions.device if positions is not None else None}, hidden_states: {hidden_states.device}, kv cache devices: {[str(kv.device) if kv is not None else None for kv in kv_caches]}, "
                        # )
            # print(f"kv cache size: {len(kv_caches)}, index is {i - self.start_layer}")
            # report_memory(f"PP rank: {get_pipeline_model_parallel_rank()} before layer {i}")
            hidden_states = layer(positions, hidden_states, kv_caches[i - self.start_layer], attn_metadata)
            # report_memory(f"PP rank: {get_pipeline_model_parallel_rank()} after layer {i}")
            # print(f"PP rank {get_pipeline_model_parallel_rank()} TP rank {get_tensor_model_parallel_rank()}: layer {i} finishes...")     
            
        if is_pipeline_model_parallel_last_rank():
            hidden_states = self.norm(hidden_states)
        else:
            # print(f"PP rank {get_pipeline_model_parallel_rank()} TP rank {get_tensor_model_parallel_rank()} about to send {hidden_states.shape}" )
            send_next_rank([hidden_states])
            # print(f"PP rank {get_pipeline_model_parallel_rank()} TP rank {get_tensor_model_parallel_rank()}, send done to {get_pipeline_model_parallel_next_rank()}, "
                #   f"shape: {hidden_states.shape}, device: {hidden_states.device}, for request with {input_ids.size()} tokens...")
        return hidden_states


class ArcticForCausalLM(nn.Module):

    def __init__(self,
                 config: ArcticConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 **kwargs) -> None:
        super().__init__()

        enable_dequantization_fusion = True
        if 'enable_dequantization_fusion' in kwargs:
            enable_dequantization_fusion = kwargs['enable_dequantization_fusion']
        config.enable_dequantization_fusion = enable_dequantization_fusion
        
        self.config = config
        self.model = ArcticModel(config, cache_config, quant_config)
        self.vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(
            self.vocab_size,
            config.hidden_size,
        )
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.unpadded_vocab_size = config.vocab_size
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # import time
        # torch.cuda.synchronize()
        # start = time.time()
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        # torch.cuda.synchronize()
        # if hasattr(self, "start_time"):
        #    duration = time.time() - self.start_time
        #    if not get_tensor_model_parallel_rank():
        #        with open(f"trace.csv", "a") as f:
        #            f.write(f"{duration},{input_ids.shape[0]}\n")
        # else:
        #    with open(f"trace.csv", "w") as f:
        #        f.write("time,tokens\n")
        # self.start_time = time.time()
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        mlp_params_mapping = []
        expert_params_mapping = []
        num_layers = self.config.num_hidden_layers

        for layer in range(num_layers):
            mlp_params_mapping.append(
                (f"layers.{layer}.residual_mlp.w13.weight",
                 f"layers.{layer}.residual_mlp.w1.weight", 0))
            mlp_params_mapping.append(
                (f"layers.{layer}.residual_mlp.w13.weight",
                 f"layers.{layer}.residual_mlp.w3.weight", 1))
            if layer % 2 == 0:
                # MLP layers
                mlp_params_mapping.append(
                    (f"layers.{layer}.block_sparse_moe.mlp.w13.weight",
                     f"layers.{layer}.block_sparse_moe.mlp.w1.weight", 0))
                mlp_params_mapping.append(
                    (f"layers.{layer}.block_sparse_moe.mlp.w13.weight",
                     f"layers.{layer}.block_sparse_moe.mlp.w3.weight", 1))
            else:
                # MoE layers
                for expert_id in range(self.config.num_local_experts):
                    expert_params_mapping.append(
                        ("ws", f"experts.{expert_id}.w1.weight", expert_id))
                    expert_params_mapping.append(
                        ("w2s", f"experts.{expert_id}.w2.weight", expert_id))
                    expert_params_mapping.append(
                        ("ws", f"experts.{expert_id}.w3.weight", expert_id))

        params_dict = dict(self.named_parameters())

        print(
            "It will take ~10 minutes loading from the 16-bit weights. "
            "Alternatively, use the prequantized 8-bit weights of arctic "
            "and set load-format to `sharded_state` will accelerate loading.")
        for name, loaded_weight in weights:
            # print(f"Global rank {torch.distributed.get_rank()}, PP rank: {get_pipeline_model_parallel_rank()}, TP rank: {get_tensor_model_parallel_rank()}, "
            #       f"Loading param {name} weight shape {loaded_weight.shape}")
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning("Skipping loading weight %s", name)
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                # print(loaded_weight, param.data, param.data.shape)
                # exit()
                break
            else:
                for param_name, weight_name, shard_id in mlp_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    try:
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id)
                    except KeyError:
                        # print(f"Global rank {torch.distributed.get_rank()}, PP rank: {get_pipeline_model_parallel_rank()}, TP rank: {get_tensor_model_parallel_rank()}, "
                        # f"Param {name} shape {loaded_weight.shape}, shard_id {shard_id} is not here.")
                        pass
                    break
                else:
                    for param_name, weight_name, shard_id \
                            in expert_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        if name not in params_dict:
                            logger.warning("Skipping loading weight %s", name)
                            break
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param,
                                      loaded_weight,
                                      weight_name,
                                      expert_id=shard_id)
                        break
                    else:
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if name not in params_dict:
                            logger.warning("Skipping loading weight %s", name)
                            continue
                        param = params_dict[name]

                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
