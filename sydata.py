from openai import AsyncOpenAI
import os
import json
import asyncio
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

load_dotenv()
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL")
)
system_prompt = """你的名字是lulu,你是一个很有创造力的助手。"""


async def generate_response(messages):
    """生成单个回复，包含推理过程"""
    response = await client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages,
    )
    content = response.choices[0].message.content
    think = response.choices[0].message.model_extra

    # 处理 think，如果不存在或为空则使用空字符串
    if think is None:
        think = ""
    elif not isinstance(think, str):
        think = str(think)

    # 组合格式：<think>think</think>content
    if think:
        return f"<think>{think}</think>{content}"
    else:
        return content


def read_jsonl(file_path):
    """读取 JSONL 文件并解析 conversations"""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "conversations" in data:
                    samples.append(data)
                else:
                    print(f"警告：第 {line_num} 行格式不正确，跳过")
            except json.JSONDecodeError as e:
                print(f"错误：第 {line_num} 行 JSON 解析失败: {e}，跳过")
    return samples


async def process_sample(sample, semaphore, pbar=None):
    """处理单个样本：遍历 conversations，识别 assistant 轮次，为每个轮次生成新回复"""
    async with semaphore:
        try:
            conversations = sample["conversations"].copy()
            new_conversations = []

            # 遍历所有对话轮次
            for i, turn in enumerate(conversations):
                if turn["role"] == "assistant":
                    # 收集该轮次之前的所有对话历史
                    history = conversations[:i]

                    # 构建消息列表
                    messages = []
                    # 检查历史中是否有 system 消息
                    has_system = any(h["role"] == "system" for h in history)

                    # 如果有 system 消息，先添加
                    if has_system:
                        for h in history:
                            if h["role"] == "system":
                                messages.append(
                                    {"role": "system", "content": h["content"]}
                                )
                                break
                    else:
                        # 如果没有 system 消息，添加默认的
                        messages.append({"role": "system", "content": system_prompt})

                    # 添加其他历史消息（按顺序添加 user 和之前的 assistant）
                    for h in history:
                        if h["role"] != "system":
                            messages.append(
                                {"role": h["role"], "content": h["content"]}
                            )

                    # 生成新回复
                    new_content = await generate_response(messages)
                    new_conversations.append(
                        {"role": "assistant", "content": new_content}
                    )
                else:
                    # 非 assistant 轮次，直接保留
                    new_conversations.append(turn)

            result = {"conversations": new_conversations}
            if pbar:
                pbar.update(1)
            return result
        except Exception as e:
            print(f"处理样本时出错: {e}")
            # 出错时返回原始样本
            if pbar:
                pbar.update(1)
            return sample


async def process_samples_concurrently(samples, max_concurrent=20):
    """并发处理所有样本"""
    semaphore = asyncio.Semaphore(max_concurrent)

    # 创建进度条
    pbar = tqdm(total=len(samples), desc="处理样本", unit="个")

    tasks = [process_sample(sample, semaphore, pbar) for sample in samples]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 关闭进度条
    pbar.close()

    # 处理异常结果
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"样本 {i+1} 处理失败: {result}，保留原始数据")
            processed_results.append(samples[i])
        else:
            processed_results.append(result)

    return processed_results


def write_jsonl(data, output_path):
    """写入新 JSONL 文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


async def main():
    """主函数：读取文件 -> 并发处理 -> 写入结果"""
    input_file = "data/sft_mini_512.jsonl"
    output_file = "data/sft_mini_512_regenerated.jsonl"

    print(f"开始读取文件: {input_file}")
    samples = read_jsonl(input_file)
    print(f"共读取 {len(samples)} 个样本")

    print("开始并发处理（最大并发数: 20）...")
    processed_samples = await process_samples_concurrently(samples, max_concurrent=20)

    print(f"开始写入结果到: {output_file}")
    write_jsonl(processed_samples, output_file)
    print("完成！")


if __name__ == "__main__":
    asyncio.run(main())
