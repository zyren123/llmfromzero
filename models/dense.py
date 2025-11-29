import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class LuluConfig(PretrainedConfig):
    model_type = "lulu"
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_gated_attention=True,
        initializer_range=0.02, # 添加初始化参数
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.use_gated_attention = use_gated_attention
        self.initializer_range = initializer_range

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def apply_rope(q, k, position_ids, theta):
    # q, k: [bs, heads, seq_len, head_dim]
    # position_ids: [bs, seq_len]
    dim = q.shape[-1]
    
    # 1. 计算频率 [dim/2]
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=q.device) / dim))
    
    # 2. 准备广播 (Broadcasting)
    # position_ids: [bs, seq_len] -> [bs, 1, seq_len, 1]
    # inv_freq: [dim/2] -> [1, 1, 1, dim/2]
    inv_freq_expanded = inv_freq[None, None, None, :]
    pos_expanded = position_ids[:, None, :, None].float()
    
    # 3. 计算角度 [bs, 1, seq_len, dim/2]
    freqs = pos_expanded * inv_freq_expanded
    
    # 4. 生成 cos/sin
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    
    # 5. 执行旋转
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    # cos, sin 自动广播到 [bs, heads, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LuluAttention(nn.Module):
    def __init__(self, config: LuluConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.use_gated_attention = config.use_gated_attention

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        
        if self.use_gated_attention:
            self.gate_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)

    def forward(self, x, position_ids, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.size()
        
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        current_cache = (k, v) if use_cache else None

        q, k = apply_rope(q, k, position_ids, self.rope_theta)

        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        # === 核心修复逻辑 ===
        # 1. Prefill 阶段 (seq_len > 1): 强制使用 is_causal=True。
        #    为了简洁，如果启用了 causal，我们暂时忽略 attention_mask (通常是全是1的padding mask)。
        #    这防止了模型变成“双向注意力”，修好了生成逻辑。
        # 2. Decoding 阶段 (seq_len == 1): 不需要 causal mask (因为只看过去)，可以使用 attention_mask。
        
        is_causal = True if seq_len > 1 else False
        
        # 如果需要 Causal Mask，为了避免 SDPA 冲突，暂时置空 attn_mask
        # (假设 batch size=1 或者右对齐 padding，这是最简方案)
        attn_mask_input = None if is_causal else attention_mask
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask_input, is_causal=is_causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        if self.use_gated_attention:
            gate = torch.sigmoid(self.gate_proj(x))
            attn_output = attn_output * gate

        return self.o_proj(attn_output), current_cache

class LuluMLP(nn.Module):
    def __init__(self, config: LuluConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class LuluBlock(nn.Module):
    def __init__(self, config: LuluConfig):
        super().__init__()
        self.self_attn = LuluAttention(config)
        self.mlp = LuluMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, position_ids, past_key_value=None, use_cache=False, attention_mask=None):
        h, cache = self.self_attn(
            self.input_layernorm(x), 
            position_ids=position_ids, 
            past_key_value=past_key_value, 
            use_cache=use_cache,
            attention_mask=attention_mask
        )
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, cache

class LuluModel(PreTrainedModel):
    config_class = LuluConfig
    _no_split_modules = ["LuluBlock"]

    def __init__(self, config: LuluConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LuluBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight 
        
        # === 核心修复: 必须调用 post_init 来初始化权重 ===
        self.post_init()

    # === 核心修复: 定义初始化规则 ===
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids, position_ids=None, attention_mask=None, past_key_values=None, use_cache=False, labels=None, **kwargs):
        bsz, seq_len = input_ids.shape
        
        if position_ids is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0)
        
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
        
        x = self.embed_tokens(input_ids)
        next_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values else None
            x, cache = layer(x, position_ids, past_key_value=layer_past, use_cache=use_cache, attention_mask=attention_mask)
            if use_cache: next_cache.append(cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=next_cache, hidden_states=None)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": True}

# Register
LuluConfig.register_for_auto_class()
LuluModel.register_for_auto_class("AutoModelForCausalLM")