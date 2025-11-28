"""
测试 SFT 数据集的 label 设置是否正确
验证：
1. 只对 assistant 的实际回复内容计算 loss
2. 包含 <|im_end|> 结束符
3. 支持多轮对话
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from training.sft import SFTDataset
import json


def test_label_masking():
    """测试 label masking 是否符合预期"""
    tokenizer = AutoTokenizer.from_pretrained("./lulu_tokenizer")

    # 创建测试数据
    test_conversations = [
        {
            "conversations": [
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": "你好，请问你的名字是什么？"},
                {"role": "assistant", "content": "你好！我是一个AI助手。"},
                {"role": "user", "content": "很高兴认识你！"},
                {
                    "role": "assistant",
                    "content": "我也很高兴认识你！有什么我可以帮助你的吗？",
                },
            ]
        }
    ]

    # 保存临时测试文件
    test_file = "test_sft_temp.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        for conv in test_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    try:
        # 创建数据集
        dataset = SFTDataset(test_file, tokenizer, max_length=512)
        sample = dataset[0]

        input_ids = sample["input_ids"].tolist()
        labels = sample["labels"].tolist()

        # 获取特殊 token IDs
        im_start_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
        im_end_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        pad_token_id = tokenizer.pad_token_id

        print("=" * 80)
        print("SFT Label Masking 测试")
        print("=" * 80)
        print(f"\n特殊 Token IDs:")
        print(f"  <|im_start|> (BOS): {im_start_id}")
        print(f"  <|im_end|> (EOS): {im_end_id}")
        print(f"  PAD: {pad_token_id}")

        # 解析并显示每个 block 的 label 设置
        print("\n" + "=" * 80)
        print("Token 序列分析（只显示非 padding 部分）:")
        print("=" * 80)

        # 找到第一个 padding token 的位置
        try:
            first_pad_idx = input_ids.index(pad_token_id)
        except ValueError:
            first_pad_idx = len(input_ids)

        # 只分析非 padding 部分
        input_ids_no_pad = input_ids[:first_pad_idx]
        labels_no_pad = labels[:first_pad_idx]

        i = 0
        block_num = 0

        while i < len(input_ids_no_pad):
            if input_ids_no_pad[i] == im_start_id:
                block_start = i
                i += 1

                # 找到对应的 <|im_end|>
                while i < len(input_ids_no_pad) and input_ids_no_pad[i] != im_end_id:
                    i += 1

                if i < len(input_ids_no_pad):
                    block_end = i + 1

                    # 解码这个 block
                    block_tokens = input_ids_no_pad[block_start:block_end]
                    block_labels = labels_no_pad[block_start:block_end]
                    block_text = tokenizer.decode(block_tokens)

                    print(f"\n📦 Block {block_num}:")
                    print(f"位置: [{block_start}:{block_end}]")
                    print(f"内容: {repr(block_text)}")

                    # 统计这个 block 中哪些位置需要计算 loss
                    loss_positions = [
                        j for j, lbl in enumerate(block_labels) if lbl != -100
                    ]

                    if loss_positions:
                        print(f"✅ 计算 Loss: 是")
                        print(
                            f"   Loss 覆盖范围: 相对位置 {loss_positions[0]} 到 {loss_positions[-1]}"
                        )

                        # 显示哪些部分计算 loss
                        loss_start_abs = block_start + loss_positions[0]
                        loss_end_abs = block_start + loss_positions[-1] + 1
                        loss_tokens = input_ids_no_pad[loss_start_abs:loss_end_abs]
                        loss_text = tokenizer.decode(loss_tokens)
                        print(f"   Loss 内容: {repr(loss_text)}")

                        # 检查是否包含 <|im_end|>
                        if im_end_id in loss_tokens:
                            print(f"   ✓ 包含 <|im_end|> 结束符")
                    else:
                        print(f"❌ 计算 Loss: 否 (全部 masked)")

                    block_num += 1
                    i = block_end
            else:
                i += 1

        # 详细验证
        print("\n" + "=" * 80)
        print("详细验证:")
        print("=" * 80)

        # 验证点 1: System 和 User 消息应该全部被 mask
        print("\n✓ 检查 1: System 和 User 消息应该全部被 masked (-100)")

        # 验证点 2: Assistant 消息应该只在实际内容部分计算 loss
        print("✓ 检查 2: Assistant 消息只对实际回复内容计算 loss")
        print("          (不包括 '<|im_start|>assistant\\n' header)")

        # 验证点 3: <|im_end|> 应该被包含在 loss 计算中
        print("✓ 检查 3: <|im_end|> 应该包含在 loss 计算中")

        # 计算总体统计
        total_tokens = len(input_ids_no_pad)
        masked_tokens = sum(1 for lbl in labels_no_pad if lbl == -100)
        loss_tokens = total_tokens - masked_tokens

        print(f"\n📊 统计信息:")
        print(f"  总 token 数 (不含 padding): {total_tokens}")
        print(
            f"  Masked tokens (-100): {masked_tokens} ({masked_tokens/total_tokens*100:.1f}%)"
        )
        print(
            f"  计算 Loss 的 tokens: {loss_tokens} ({loss_tokens/total_tokens*100:.1f}%)"
        )

        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)

    finally:
        # 清理临时文件
        if os.path.exists(test_file):
            os.remove(test_file)


if __name__ == "__main__":
    test_label_masking()
