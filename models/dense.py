import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    GenerationMixin,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
import math


class LuluConfig(PretrainedConfig):
    model_type = "lulu"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=4,  # GQA
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        output_attentions=False,
        output_hidden_states=False,
        # use_return_dict=True,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # use_return_dict=use_return_dict,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
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

        # Build here to make `forward` faster
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
        # Different from paper, but common in HF implementations (cat along last dim)
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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    # cos, sin: [max_seq_len, dim] or [seq_len, dim]
    # q, k: [bs, heads, seq_len, dim]
    # position_ids: [bs, seq_len] (optional)
    if position_ids is not None:
        # position_ids: [bs, seq_len]
        # Select cos and sin based on position_ids
        # cos[position_ids]: [bs, seq_len, dim]
        cos_selected = cos[position_ids]  # [bs, seq_len, dim]
        sin_selected = sin[position_ids]  # [bs, seq_len, dim]
        # Expand to match q, k dimensions: [bs, 1, seq_len, dim]
        cos = cos_selected.unsqueeze(1)
        sin = sin_selected.unsqueeze(1)
    else:
        # Use sequential positions (0, 1, 2, ...)
        # cos, sin: [seq_len, dim] -> [1, 1, seq_len, dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LuluAttention(nn.Module):
    def __init__(self, config: LuluConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
    ):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache
        if past_key_value is not None:
            # past_key_value: tuple of (k_cache, v_cache) with shape [bs, num_heads, cache_len, head_dim]
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Determine sequence length for RoPE
        kv_seq_len = k.shape[2]

        # Prepare position_ids for RoPE
        if position_ids is not None:
            # position_ids provided: typically [bs, q_len] for new tokens
            if past_key_value is not None and position_ids.shape[-1] == q_len:
                # Only new tokens provided, need full position_ids for k
                cache_len = past_k.shape[2]
                cache_position_ids = (
                    torch.arange(0, cache_len, device=position_ids.device)
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )
                position_ids_k = torch.cat([cache_position_ids, position_ids], dim=1)
                position_ids_q = position_ids
            else:
                # Assume full sequence or no cache
                position_ids_q = position_ids
                position_ids_k = position_ids
        else:
            # Generate position_ids if not provided
            if past_key_value is not None:
                cache_len = past_k.shape[2]
                position_ids_q = (
                    torch.arange(cache_len, cache_len + q_len, device=k.device)
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )
                cache_position_ids = (
                    torch.arange(0, cache_len, device=k.device)
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )
                position_ids_k = torch.cat([cache_position_ids, position_ids_q], dim=1)
            else:
                position_ids_q = (
                    torch.arange(0, q_len, device=k.device).unsqueeze(0).expand(bsz, -1)
                )
                position_ids_k = position_ids_q

        # Apply RoPE
        # FIXED: Calculate max position needed to ensure cache is big enough to be indexed by position_ids
        if position_ids is not None:
            # We need the cache to cover the largest index in position_ids
            max_pos = max(position_ids_q.max().item(), position_ids_k.max().item()) + 1
        else:
            max_pos = max(q_len, kv_seq_len)

        # Get the full needed cos/sin cache
        cos, sin = self.rotary_emb(v, seq_len=max_pos)

        # Apply RoPE with appropriate position_ids
        # Note: apply_rotary_pos_emb expects cos/sin to be indexable by position_ids
        q, _ = apply_rotary_pos_emb(q, q, cos, sin, position_ids=position_ids_q)
        _, k = apply_rotary_pos_emb(k, k, cos, sin, position_ids=position_ids_k)

        # Repeat KV for GQA (before attention computation)
        if self.num_key_value_groups > 1:
            k_repeated = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v_repeated = v.repeat_interleave(self.num_key_value_groups, dim=1)
        else:
            k_repeated = k
            v_repeated = v

        attn_weights = torch.matmul(q, k_repeated.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v_repeated)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(bsz, q_len, self.hidden_size)
        )
        output = self.o_proj(attn_output)

        # Return cache if use_cache is True
        if use_cache:
            # Return k, v before GQA repetition (original num_key_value_heads)
            return output, (k, v)
        return output, None


class LuluMLP(nn.Module):
    def __init__(self, config: LuluConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LuluBlock(nn.Module):
    def __init__(self, config: LuluConfig):
        super().__init__()
        self.self_attn = LuluAttention(config)
        self.mlp = LuluMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if use_cache:
            return hidden_states, present_key_value
        return hidden_states


class LuluModel(PreTrainedModel, GenerationMixin):
    config_class = LuluConfig

    def __init__(self, config: LuluConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LuluBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

        self.post_init()

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

    # ------------------------------------------------------------------
    # ADDED METHODS FOR GENERATION COMPATIBILITY
    # ------------------------------------------------------------------
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            # If we have cache, we only need the last input token
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # Create position_ids on the fly for batching
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                # If we have cache, we only need the position of the last token
                position_ids = position_ids[:, -1].unsqueeze(-1)

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        return model_inputs

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        **kwargs,
    ):
        # Set defaults from config
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Get input embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            hidden_states = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
            hidden_states = inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Generate position_ids if not provided
        if position_ids is None:
            if past_key_values is not None:
                # During generation, position_ids should be the position of the new tokens
                # past_key_values[0][0] is the k cache from first layer, shape [bs, heads, cache_len, head_dim]
                cache_len = past_key_values[0][0].shape[2]
                position_ids = (
                    torch.arange(
                        cache_len, cache_len + seq_length, device=hidden_states.device
                    )
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
            else:
                position_ids = (
                    torch.arange(0, seq_length, device=hidden_states.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )

        # Create causal mask
        if attention_mask is None:
            if past_key_values is not None:
                # During generation with cache, only need mask for new tokens
                # Create a causal mask for the new sequence length
                mask = torch.full(
                    (seq_length, seq_length), float("-inf"), device=hidden_states.device
                )
                mask = torch.triu(mask, diagonal=1)
                attention_mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
            else:
                # Full causal mask
                mask = torch.full(
                    (seq_length, seq_length), float("-inf"), device=hidden_states.device
                )
                mask = torch.triu(mask, diagonal=1)
                attention_mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        else:
            # If attention_mask is provided, ensure it has the right shape
            if attention_mask.dim() == 2:
                # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                # [batch_size, 1, seq_len] -> [batch_size, 1, 1, seq_len]
                attention_mask = attention_mask[:, None, :, :]

            # FIXED: Replace 1s with 0s and 0s with -inf safely without using arithmetic that causes NaN
            # Old error code: attention_mask = (1.0 - attention_mask) * float("-inf")

            # Create a new mask initialized to 0s
            expanded_mask = torch.zeros_like(attention_mask, dtype=hidden_states.dtype)
            # Fill the positions where the original mask was 0 with -inf
            expanded_mask.masked_fill_(attention_mask == 0, float("-inf"))
            attention_mask = expanded_mask

        # Initialize past_key_values if None
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # Store hidden states if needed
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # Process through layers
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0] if use_cache else layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

            if output_attentions:
                # Note: We don't return attentions from LuluAttention currently
                all_self_attns += (None,)

        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,)
            if use_cache:
                output += (next_decoder_cache,)
            if output_hidden_states:
                output += (all_hidden_states,)
            if output_attentions:
                output += (all_self_attns,)
            return output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_decoder_cache if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# Register for AutoModel
LuluConfig.register_for_auto_class()
LuluModel.register_for_auto_class("AutoModelForCausalLM")
