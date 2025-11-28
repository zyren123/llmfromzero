import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    GenerationMixin,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from .dense import LuluConfig, LuluModel


class VisionConfig(PretrainedConfig):
    model_type = "custom_vision"

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=1536,
        hidden_act="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act


class LuluVLConfig(PretrainedConfig):
    model_type = "lulu_vl"

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        projector_hidden_act="gelu",
        tie_word_embeddings=True,
        use_cache=True,
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
        if vision_config is None:
            self.vision_config = VisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = VisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        if text_config is None:
            # Default to a smaller Dense config for VLM to keep total size manageable
            self.text_config = LuluConfig(
                vocab_size=32000,
                hidden_size=512,
                intermediate_size=2048,
                num_hidden_layers=6,
                num_attention_heads=16,
                num_key_value_heads=4,
            )
        elif isinstance(text_config, dict):
            self.text_config = LuluConfig(**text_config)
        else:
            self.text_config = text_config

        self.projector_hidden_act = projector_hidden_act
        self.use_cache = use_cache


class VisionEncoder(nn.Module):
    """
    Simple ViT-style encoder
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            activation=config.hidden_act,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_hidden_layers
        )

    def forward(self, pixel_values):
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.encoder(x)
        return x


class LuluVLModel(PreTrainedModel, GenerationMixin):
    config_class = LuluVLConfig

    def __init__(self, config: LuluVLConfig):
        super().__init__(config)
        self.vision_tower = VisionEncoder(config.vision_config)
        self.language_model = LuluModel(config.text_config)

        self.projector = nn.Sequential(
            nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size),
            nn.GELU(),
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size),
        )

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
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
            batch_size, text_seq_length = input_ids.shape
            text_embeds = self.language_model.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, text_seq_length = inputs_embeds.shape[:2]
            text_embeds = inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Handle image features
        image_seq_length = 0
        if pixel_values is not None:
            image_features = self.vision_tower(pixel_values)
            image_features = self.projector(image_features)
            image_seq_length = image_features.shape[1]
            inputs_embeds = torch.cat([image_features, text_embeds], dim=1)
        else:
            inputs_embeds = text_embeds

        total_seq_length = inputs_embeds.shape[1]

        # Handle attention mask
        if attention_mask is not None:
            if pixel_values is not None:
                # Add mask for image tokens (all 1s)
                image_mask = torch.ones(
                    (batch_size, image_seq_length),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([image_mask, attention_mask], dim=1)
        else:
            # Create default attention mask
            attention_mask = torch.ones(
                (batch_size, total_seq_length), device=inputs_embeds.device
            )

        # Handle labels
        if labels is not None and pixel_values is not None:
            image_labels = torch.full(
                (batch_size, image_seq_length),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            labels = torch.cat([image_labels, labels], dim=1)

        # Generate position_ids if not provided
        if position_ids is None:
            if past_key_values is not None:
                # During generation, position_ids should be the position of the new tokens
                if past_key_values[0] is not None:
                    cache_len = past_key_values[0][0].shape[2]
                    # Image tokens are always at positions 0 to image_seq_length-1
                    # Text tokens continue from image_seq_length
                    if pixel_values is not None:
                        # First forward: image tokens + text tokens
                        # Subsequent forwards: only new text tokens
                        if cache_len >= image_seq_length:
                            # We already have image tokens in cache, only add new text positions
                            position_ids = (
                                torch.arange(
                                    cache_len,
                                    cache_len + text_seq_length,
                                    device=inputs_embeds.device,
                                )
                                .unsqueeze(0)
                                .expand(batch_size, -1)
                            )
                        else:
                            # First forward with images
                            image_positions = torch.arange(
                                0, image_seq_length, device=inputs_embeds.device
                            )
                            text_positions = torch.arange(
                                image_seq_length,
                                image_seq_length + text_seq_length,
                                device=inputs_embeds.device,
                            )
                            position_ids = (
                                torch.cat([image_positions, text_positions], dim=0)
                                .unsqueeze(0)
                                .expand(batch_size, -1)
                            )
                    else:
                        position_ids = (
                            torch.arange(
                                cache_len,
                                cache_len + text_seq_length,
                                device=inputs_embeds.device,
                            )
                            .unsqueeze(0)
                            .expand(batch_size, -1)
                        )
                else:
                    # No cache yet
                    if pixel_values is not None:
                        image_positions = torch.arange(
                            0, image_seq_length, device=inputs_embeds.device
                        )
                        text_positions = torch.arange(
                            image_seq_length,
                            image_seq_length + text_seq_length,
                            device=inputs_embeds.device,
                        )
                        position_ids = (
                            torch.cat([image_positions, text_positions], dim=0)
                            .unsqueeze(0)
                            .expand(batch_size, -1)
                        )
                    else:
                        position_ids = (
                            torch.arange(
                                0, text_seq_length, device=inputs_embeds.device
                            )
                            .unsqueeze(0)
                            .expand(batch_size, -1)
                        )
            else:
                # No cache
                if pixel_values is not None:
                    image_positions = torch.arange(
                        0, image_seq_length, device=inputs_embeds.device
                    )
                    text_positions = torch.arange(
                        image_seq_length,
                        image_seq_length + text_seq_length,
                        device=inputs_embeds.device,
                    )
                    position_ids = (
                        torch.cat([image_positions, text_positions], dim=0)
                        .unsqueeze(0)
                        .expand(batch_size, -1)
                    )
                else:
                    position_ids = (
                        torch.arange(0, text_seq_length, device=inputs_embeds.device)
                        .unsqueeze(0)
                        .expand(batch_size, -1)
                    )

        # Create causal mask
        seq_len = inputs_embeds.size(1)
        if attention_mask.dim() == 2:
            # Convert to attention mask format
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask[:, None, None, :]
            # Convert 0/1 mask to -inf/0 mask
            attention_mask = (1.0 - attention_mask) * float("-inf")
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask[:, None, :, :]
            attention_mask = (1.0 - attention_mask) * float("-inf")
        else:
            # Create causal mask
            if past_key_values is not None:
                # During generation with cache, only need mask for new tokens
                mask = torch.full(
                    (seq_len, seq_len), float("-inf"), device=inputs_embeds.device
                )
                mask = torch.triu(mask, diagonal=1)
                attention_mask = mask.unsqueeze(0).unsqueeze(0)
            else:
                mask = torch.full(
                    (seq_len, seq_len), float("-inf"), device=inputs_embeds.device
                )
                mask = torch.triu(mask, diagonal=1)
                attention_mask = mask.unsqueeze(0).unsqueeze(0)

        # Initialize past_key_values if None
        if past_key_values is None:
            past_key_values = [None] * len(self.language_model.layers)

        # Store hidden states if needed
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        hidden_states = inputs_embeds

        # Process through layers
        for idx, decoder_layer in enumerate(self.language_model.layers):
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
                all_self_attns += (None,)

        hidden_states = self.language_model.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        logits = self.language_model.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.text_config.vocab_size),
                shift_labels.view(-1),
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
LuluVLConfig.register_for_auto_class()
LuluVLModel.register_for_auto_class("AutoModelForCausalLM")
