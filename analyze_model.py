import torch
import torch.nn as nn
from models.dense import DenseConfig, DenseModel
from models.moe import DeepSeekV3Config, DeepSeekV3Model, DeepSeekMoE
from models.vlm import VLMConfig, VLMModel, VisionConfig


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(num):
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


def analyze_dense(config=None):
    print("\n" + "=" * 50)
    print("Analyzing Dense Model (Qwen2.5 Style)")
    print("=" * 50)
    if config is None:
        config = DenseConfig()  # Use defaults (50M target)

    model = DenseModel(config)
    total_params = count_parameters(model)

    print(
        f"Config: Layers={config.num_hidden_layers}, Hidden={config.hidden_size}, Heads={config.num_attention_heads}, KV_Heads={config.num_key_value_heads}"
    )
    print(f"Total Parameters: {format_params(total_params)}")


def analyze_vlm(config=None):
    print("\n" + "=" * 50)
    print("Analyzing VLM (Vision + Projector + LLM)")
    print("=" * 50)
    if config is None:
        config = VLMConfig(VisionConfig(), DenseConfig())  # Use defaults (50M target)

    model = VLMModel(config)

    total_params = count_parameters(model)
    vision_params = count_parameters(model.vision_tower)
    projector_params = count_parameters(model.projector)
    llm_params = count_parameters(model.language_model)

    print(f"Total Parameters: {format_params(total_params)}")
    print("-" * 30)
    print(
        f"Vision Tower:   {format_params(vision_params):>10} ({vision_params / total_params:.2%})"
    )
    print(
        f"Projector:      {format_params(projector_params):>10} ({projector_params / total_params:.2%})"
    )
    print(
        f"Language Model: {format_params(llm_params):>10} ({llm_params / total_params:.2%})"
    )


def analyze_moe(config=None):
    print("\n" + "=" * 50)
    print("Analyzing MoE Model (DeepSeek-V3 Style)")
    print("=" * 50)
    if config is None:
        config = DeepSeekV3Config()  # Use defaults (50M target)

    model = DeepSeekV3Model(config)
    total_params = count_parameters(model)

    total_routed_params = 0
    single_expert_params = 0

    for name, module in model.named_modules():
        if isinstance(module, DeepSeekMoE):
            if len(module.routed_experts) > 0:
                one_expert = module.routed_experts[0]
                p_expert = count_parameters(one_expert)
                single_expert_params += p_expert
                total_routed_params += sum(
                    count_parameters(e) for e in module.routed_experts
                )

    active_routed_params = single_expert_params * config.moe_top_k
    active_params = total_params - total_routed_params + active_routed_params

    print(
        f"Config: Experts={config.moe_num_experts}, Shared={config.moe_num_shared_experts}, TopK={config.moe_top_k}"
    )
    print("-" * 30)
    print(f"Total Parameters:  {format_params(total_params):>10}")
    print(f"Active Parameters: {format_params(active_params):>10}")
    print(f"Sparse Ratio:      {active_params / total_params:.2%}")
    print("-" * 30)
    print(f"Total Routed Params: {format_params(total_routed_params)}")
    print(f"Active Routed Params: {format_params(active_routed_params)}")


if __name__ == "__main__":
    analyze_dense()
    analyze_moe()
    analyze_vlm()
