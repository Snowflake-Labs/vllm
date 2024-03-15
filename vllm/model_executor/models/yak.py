"""Inference-only Yak model."""
from typing import List, Optional, Tuple

import torch
from torch import nn

# TODO(Hao): this assume it will depend on the transformer version which includes Yak
from transformers import YakConfig
from transformers.activitions import ACT2FN

from vllm.config import LoRAConfig
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.sequence import SamplerOutput
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)

KVCache = Tuple[torch.Tensor, torch.Tensor]


class YakMLP(nn.Module):
    def __init__(self, config, layer_id, expert_id=-1, is_residual_mlp=False):
        super(YakMLP).__init__()
        self.hidden_dim = config.hidden_size
        self.expert_id = expert_id
        self.layer_id = layer_id

        # TODO(Hao): make this tensor-parallel using RowParallelLinear
        self.ffn_dim = config.intermediate_size if not is_redisual_mlp else self.hidden_dim

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim , bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class YakMoE(nn.Module):
    """Model-parallel implementation of Yak MoE Layer.

    Note that Yak has *every other* layer to be an MoE (different from Mixtral).
    """
    def __init__(self, config: YakConfig,
                 layer_id: int,
                 tp_size: Optional[int] = None,
                 params_dtype: Optional[torch.dtype] = None):
        super(YakMoE).__init__()

        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.hidden_dim = config.hidden_dim
        self.num_total_experts = config.num_experts
        self.layer_id = layer_id
        self.top_k = config.num_experts_per_tok
        self.intermediate_dim = self.hidden_dim // self.tp_size

        self.is_moe_layer = (layer_id+1) % config.moe_layer_frequency == 0

        # Some other parameters
        self.params_dtype = None

        if not self.is_moe_layer:
            self.mlp = YakMLP(config, layer_id=layer_id)
        else:
            self.gate = ReplicatedLinear(self.hidden_dim,
                                         self.num_total_experts,
                                         bias=False,
                                         params_dtype=self.params_dtype,
                                         linear_method=None)
            self.experts = nn.ModuleList(
                [YakMLP(config, layer_id=layer_id, expert_id=i) for i in range(self.num_experts)])

    # Copied from transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock.forward with Mixtral->Yak
    def local_moe(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.top_k > 1:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        # routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
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
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor):
        if self.is_moe_layer:
            return self.local_moe(hidden_states)
        else:
            return self.mlp(hidden_states)


class YakRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        YakRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class YakRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class YakAttention(nn.Module):
    def __init__(self, config: YakConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # TODO(Hao): finish this for TP
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads // config.tp_size * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads // config.tp_size * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads // config.tp_size * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads // config.tp_size * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = YakRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_key_value_heads
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output = self.o_proj(attn_output)
        return output

class YakDecoderLayer(nn.Module):
    def __init__(
        self,
        config: YakConfig,
        layer_idx: int,
        linear_method: Optional[LinearM] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = YakAttention(config, layer_idx)
        self.block_sparse_moe = YakMoE(config, layer_id=layer_idx)

        # TODO(Hao): reconsider the YakRMSNorm here.
        self.input_layernorm = YakRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_residual = config.use_residual and self.block_sparse_moe.is_moe_layer
        if self.use_residual:
            self.residual_layernorm = YakRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.residual_mlp = YakMLP(config, layer_id=layer_idx, is_residual_mlp=True)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        residual_input = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # self attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata
        )
        hidden_states = residual_input + hidden_states

        residual_attn = hidden_states
        if self.use_residual:
            # residual mlp part
            hidden_states = self.residual_layernorm(hidden_states)
            hidden_states = self.residual_mlp(hidden_states)
            residual_residual = residual_attn + hidden_states
            # residual mlp moe part
            hidden_states = self.post_attention_layernorm(residual_input)
            hidden_states = self.block_sparse_moe(hidden_states)
            hidden_states = residual_residual + hidden_states
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

        # change this to VocabParallelEmbedding()
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            YakDecoderLayer(config, layer_idx, linear_method=linear_method)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self._attn_implementation = config._attn_implementation
        #TODO(Hao): vllm has its own impl or RMSNorm, should we consider?
        self.norm = YakRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(positions, hidden_states,
                                  kv_caches[i], input_metadata)
            hidden_states = self.norm(hidden_states)
            return hidden_states


class YakForCausalLM(YakPretrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = YakModel(config)
        self.vocab_size = config.vocab_size

        # TODO(Hao): change this with ParallelLMHead
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.unpadded_vocab_size = config.vocab_size
        # Initialize weights and apply final processing
        self.post_init()
        self.sampler = Sampler(self.unpadded_vocab_size, config.vocab_size)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[KVCache],
                input_metadata: InputMetadata,
            ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata)
        return hidden_states

    def sample(
        self,
        hidden_states: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.llm_head.weight, hidden_states, sampling_metadata)
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

        expert_params_mapping = [
            # (param_name, weight_name, expert_id)
            ("ws" if weight_name in ["w1", "w3"] else "w2s",
             f"experts.{expert_id}.{weight_name}.weight", expert_id)
            for expert_id in range(self.config.num_local_experts)
            for weight_name in ["w1", "w2", "w3"]
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path,
                cache_dir,
                load_format,
                revision,
                fall_back_to_pt=False):
            if "rotary_emb.inv_freq" in name:
                continue

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
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  weight_name,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
