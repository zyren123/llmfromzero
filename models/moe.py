import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
import math


class DeepSeekV3Config(PretrainedConfig):
    model_type = "deepseek_v3"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=384,
        num_hidden_layers=4,
        num_attention_heads=12,
        num_key_value_heads=4,  # GQA
        moe_num_experts=8,
        moe_num_shared_experts=1,
        moe_top_k=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.moe_num_experts = moe_num_experts
        self.moe_num_shared_experts = moe_num_shared_experts
        self.moe_top_k = moe_top_k
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MoEAttention(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(self, hidden_states, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(v, seq_len=q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat KV for GQA
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(bsz, q_len, self.hidden_size)
        )
        return self.o_proj(attn_output)


class DeepSeekMoE(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.num_shared_experts = config.moe_num_shared_experts
        self.top_k = config.moe_top_k
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.hidden_size * 4  # Standard expansion

        self.shared_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
                    nn.SiLU(),
                    nn.Linear(self.intermediate_size, self.hidden_size, bias=False),
                )
                for _ in range(self.num_shared_experts)
            ]
        )

        self.routed_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
                    nn.SiLU(),
                    nn.Linear(self.intermediate_size, self.hidden_size, bias=False),
                )
                for _ in range(self.num_experts)
            ]
        )

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

    def forward(self, hidden_states):
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        shared_output = sum(
            expert(hidden_states_flat) for expert in self.shared_experts
        )

        router_logits = self.router(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        final_routed_output = torch.zeros_like(hidden_states_flat)

        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]
            expert_weights = top_k_weights[:, k].unsqueeze(-1)

            for expert_idx in range(self.num_experts):
                mask = expert_indices == expert_idx
                if mask.any():
                    expert_input = hidden_states_flat[mask]
                    expert_out = self.routed_experts[expert_idx](expert_input)
                    final_routed_output[mask] += expert_out * expert_weights[mask]

        total_output = shared_output + final_routed_output
        return total_output.view(bsz, seq_len, hidden_dim)


class DeepSeekBlock(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.self_attn = MoEAttention(config)
        self.moe = DeepSeekMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DeepSeekV3Model(PreTrainedModel, GenerationMixin):
    config_class = DeepSeekV3Config

    def __init__(self, config: DeepSeekV3Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepSeekBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.embed_tokens(input_ids)

        seq_len = input_ids.size(1)
        if attention_mask is None:
            mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=input_ids.device
            )
            mask = torch.triu(mask, diagonal=1)
            attention_mask = mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )
