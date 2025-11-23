# 从零开始训练大语言模型 (Educational LLM Project)

本项目实现了一个完整的从零开始训练大语言模型 (LLM) 的流程，旨在用于教学目的。项目涵盖了 Tokenizer 训练、多种模型架构实现以及不同的训练阶段。

## 1. 项目结构

```
llm-from-scratch/
├── tokenizer/
│   └── train_tokenizer.py  # BPE Tokenizer 训练脚本
├── models/
│   ├── dense.py            # Qwen2.5 风格的 Dense 模型
│   ├── moe.py              # DeepSeek-V3 风格的 MoE 模型 (MLA + DeepSeekMoE)
│   └── vlm.py              # 视觉语言模型 (ViT + Projector + LLM)
├── training/
│   ├── pretrain.py         # 因果语言模型预训练循环 (Pretrain)
│   ├── sft.py              # 监督微调循环 (SFT/指令微调)
│   └── dpo.py              # 直接偏好优化循环 (DPO)
└── verify_project.py       # 端到端验证脚本
```

## 2. 已实现的核心功能

### Tokenizer (分词器)
- **BPE 训练**: 使用 `tokenizers` 库从原始文本训练 Byte-Pair Encoding 分词器。
- **Transformers 兼容性**: 将训练好的 tokenizer 封装为 `PreTrainedTokenizerFast`，以便与 Hugging Face `transformers` 无缝集成。

### 模型架构
- **Dense (Qwen2.5 风格)**:
  - 实现了 RMSNorm, SwiGLU, 和 RoPE (旋转位置编码)。
  - 支持分组查询注意力 (GQA)。
- **MoE (DeepSeek-V3 风格)**:
  - **MLA (多头潜在注意力)**: 实现了低秩 KV 压缩以减少显存占用。
  - **DeepSeekMoE**: 实现了细粒度专家分割，包含共享专家 (Shared Experts) 和 路由专家 (Routed Experts)。
- **VLM (视觉语言模型)**:
  - 结合了简单的视觉编码器 (ViT 风格) 和 Dense LLM。
  - 使用 MLP 投影层将图像特征映射到文本嵌入空间。

### 训练循环
- **Pretrain (预训练)**: 基于原始文本文件的标准 Next-token prediction 任务。
- **SFT (指令微调)**: 支持对用户提示 (Prompt) 进行 Mask 操作，仅计算助手回复部分的 Loss。
- **DPO (直接偏好优化)**: 使用 Chosen/Rejected 配对数据进行偏好对齐，手动实现了 DPO Loss。

### Think Mode (思考模式)
- **特殊 Token**: Tokenizer 支持 `<think>` 和 `</think>` 标签。
- **推理展示**: `inference.py` 演示了如何解析并单独展示“思考过程” (Thinking Process) 与最终答案，模拟推理模型的输出效果。

## 3. 验证

项目提供了一个 `verify_project.py` 脚本来测试整个流程。它会：
1. 为所有阶段生成伪数据。
2. 训练一个 Tokenizer。
3. 实例化所有 3 种模型，并运行 3 种训练模式的训练步骤。

运行验证脚本：
```bash
python verify_project.py
```
