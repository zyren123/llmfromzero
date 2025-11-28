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
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
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

    def forward(self, input_ids, pixel_values=None, attention_mask=None, labels=None):
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is not None:
            image_features = self.vision_tower(pixel_values)
            image_features = self.projector(image_features)
            inputs_embeds = torch.cat([image_features, inputs_embeds], dim=1)

            if attention_mask is not None:
                image_mask = torch.ones(
                    image_features.shape[:2], device=attention_mask.device
                )
                attention_mask = torch.cat([image_mask, attention_mask], dim=1)

            if labels is not None:
                image_labels = torch.full(
                    image_features.shape[:2],
                    -100,
                    device=labels.device,
                    dtype=labels.dtype,
                )
                labels = torch.cat([image_labels, labels], dim=1)

        hidden_states = inputs_embeds

        seq_len = hidden_states.size(1)
        if attention_mask is None:
            mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=hidden_states.device
            )
            mask = torch.triu(mask, diagonal=1)
            attention_mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            # Simplified mask handling for demo
            mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=hidden_states.device
            )
            mask = torch.triu(mask, diagonal=1)
            attention_mask = mask.unsqueeze(0).unsqueeze(0)

        for layer in self.language_model.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.language_model.norm(hidden_states)
        logits = self.language_model.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.text_config.vocab_size),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )


# Register for AutoModel
LuluVLConfig.register_for_auto_class()
LuluVLModel.register_for_auto_class("AutoModelForCausalLM")
