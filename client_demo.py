import re
import json
from openai import OpenAI

# 假设你已经用 vLLM 启动了模型：
# vllm serve /path/to/your/model --served-model-name lulu-1.0

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="empty",  # vLLM 本地部署通常不需要 key
)


def parse_response(content):
    """
    解析模型返回的原始内容，提取 think 和 tool_calls
    """
    result = {"think": None, "tool_calls": [], "content": content}

    # 1. 提取 <think>
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, content, re.DOTALL)
    if think_match:
        result["think"] = think_match.group(1).strip()
        # 从 content 中移除 think 部分，或者保留取决于你的 UI 需求
        content = re.sub(think_pattern, "", content, flags=re.DOTALL).strip()

    # 2. 提取 <tool_call>
    # 假设模型输出格式为 <tool_call>{"name": "...", "arguments": ...}</tool_call>
    tool_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_matches = re.finditer(tool_pattern, content, re.DOTALL)

    for match in tool_matches:
        try:
            tool_json = json.loads(match.group(1))
            # 构造 OpenAI 风格的 tool_call 对象
            result["tool_calls"].append(
                {
                    "id": "call_" + str(len(result["tool_calls"])),  # 生成一个伪 ID
                    "type": "function",
                    "function": {
                        "name": tool_json.get("name"),
                        "arguments": json.dumps(tool_json.get("arguments")),
                    },
                }
            )
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse tool call JSON: {match.group(1)}")

    # 如果有 tool_calls，通常 content 里的 tool_call 标签可以清理掉
    if result["tool_calls"]:
        content = re.sub(tool_pattern, "", content, flags=re.DOTALL).strip()

    result["content"] = content
    return result


def chat_with_model():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "帮我查一下新加坡的天气，顺便思考一下我为什么问这个问题。",
        },
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    print("Sending request to vLLM...")
    response = client.chat.completions.create(
        model="lulu-1.0", messages=messages, tools=tools, temperature=0.7
    )

    raw_content = response.choices[0].message.content
    print(f"\n[Raw Response]:\n{raw_content}\n")

    # 手动解析（这是从零训练模型的必经之路）
    parsed = parse_response(raw_content)

    print("-" * 30)
    if parsed["think"]:
        print(f"\n[Thinking Process]:\n{parsed['think']}")

    if parsed["tool_calls"]:
        print(f"\n[Tool Calls Detected]:")
        for tc in parsed["tool_calls"]:
            print(f"  Function: {tc['function']['name']}")
            print(f"  Args:     {tc['function']['arguments']}")

    if parsed["content"]:
        print(f"\n[Final Answer]:\n{parsed['content']}")


if __name__ == "__main__":
    # 注意：这需要真实的 vLLM 服务运行才能跑通
    # 这里只是代码演示
    print("This script demonstrates how to parse custom model outputs.")
    print("Please ensure vLLM is running locally.")
    # chat_with_model()
