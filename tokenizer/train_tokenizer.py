import os
import json
import sys

# Add parent directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast
from utils.logger import Logger
import argparse


# def batch_iterator(files, logger=None):
#     """
#     Yields text data from files (supports .txt and .jsonl).
#     """
#     for file_path in files:
#         if not os.path.exists(file_path):
#             if logger:
#                 logger.warning(f"File {file_path} not found.")
#             else:
#                 print(f"Warning: File {file_path} not found.")
#             continue


#         with open(file_path, "r", encoding="utf-8") as f:
#             if file_path.endswith(".jsonl"):
#                 for line in f:
#                     if not line.strip():
#                         continue
#                     try:
#                         data = json.loads(line)
#                         if "text" in data:
#                             yield data["text"]
#                     except json.JSONDecodeError:
#                         if logger:
#                             logger.warning(f"Failed to decode JSON line in {file_path}")
#                         else:
#                             print(f"Warning: Failed to decode JSON line in {file_path}")
#             else:
#                 # Assume plain text
#                 for line in f:
#                     yield line
def batch_iterator(
    files, logger=None, max_bytes=1 * 1024 * 1024 * 1024
):  # 默认限制 1GB
    """
    Yields text data from files (supports .txt and .jsonl).
    Stops after processing max_bytes of data.
    """
    current_bytes = 0

    if logger:
        logger.info(f"Data limit set to: {max_bytes / (1024*1024):.2f} MB")
    else:
        print(f"Data limit set to: {max_bytes / (1024*1024):.2f} MB")

    for file_path in files:
        # 如果已经达到总限制，就不再打开新文件
        if current_bytes >= max_bytes:
            break

        if not os.path.exists(file_path):
            msg = f"File {file_path} not found."
            if logger:
                logger.warning(msg)
            else:
                print(f"Warning: {msg}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".jsonl"):
                for line in f:
                    # 检查是否超限
                    if current_bytes >= max_bytes:
                        break

                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if "text" in data:
                            text = data["text"]
                            text_len = len(text.encode("utf-8"))  # 计算字节大小

                            yield text
                            current_bytes += text_len
                    except json.JSONDecodeError:
                        if logger:
                            logger.warning(f"Failed to decode JSON line in {file_path}")
            else:
                # Assume plain text
                for line in f:
                    if current_bytes >= max_bytes:
                        break

                    line_len = len(line.encode("utf-8"))
                    yield line
                    current_bytes += line_len

    msg = f"Tokenizer training completed. Processed {current_bytes / (1024*1024):.2f} MB of data."
    if logger:
        logger.info(msg)
    else:
        print(msg)


def get_chat_template():
    """
    Returns the Jinja2 template for Lulu (Qwen-like) with support for tools, think, and image.
    """
    template = (
        "{%- if tools %}"
        "{{- '<|im_start|>system\\n' }}"
        "{%- if messages[0]['role'] == 'system' %}"
        "{{- messages[0]['content'] }}"
        "{%- else %}"
        "{{- 'You are Lulu, created by Lulu AI. You are a helpful assistant.' }}"
        "{%- endif %}"
        '{{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}'
        "{%- for tool in tools %}"
        '{{- "\\n" }}'
        "{{- tool | tojson }}"
        "{%- endfor %}"
        '{{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}'
        "{%- else %}"
        "{%- if messages[0]['role'] == 'system' %}"
        "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
        "{%- else %}"
        "{{- '<|im_start|>system\\nYou are Lulu, created by Lulu AI. You are a helpful assistant.<|im_end|>\\n' }}"
        "{%- endif %}"
        "{%- endif %}"
        "{%- for message in messages %}"
        '{%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}'
        "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
        '{%- elif message.role == "assistant" %}'
        "{{- '<|im_start|>' + message.role }}"
        "{%- if message.content %}"
        "{{- '\\n' + message.content }}"
        "{%- endif %}"
        "{%- for tool_call in message.tool_calls %}"
        "{%- if tool_call.function is defined %}"
        "{%- set tool_call = tool_call.function %}"
        "{%- endif %}"
        '{{- \'\\n<tool_call>\\n{"name": "\' }}'
        "{{- tool_call.name }}"
        '{{- \'", "arguments": \' }}'
        "{{- tool_call.arguments | tojson }}"
        "{{- '}\\n</tool_call>' }}"
        "{%- endfor %}"
        "{{- '<|im_end|>\\n' }}"
        '{%- elif message.role == "tool" %}'
        '{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}'
        "{{- '<|im_start|>user' }}"
        "{%- endif %}"
        "{{- '\\n<tool_response>\\n' }}"
        "{{- message.content }}"
        "{{- '\\n</tool_response>' }}"
        '{%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}'
        "{{- '<|im_end|>\\n' }}"
        "{%- endif %}"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|im_start|>assistant\\n' }}"
        "{%- endif %}"
    )
    return template


def train_tokenizer(
    files,
    vocab_size=32000,
    save_path="lulu_tokenizer",
    max_train_bytes=1024 * 1024 * 1024,
):
    """
    Trains a BPE tokenizer from scratch with Tool Call, Think, and Image support.
    """
    logger = Logger("train_tokenizer")
    logger.info(f"Initializing tokenizer training with vocab_size={vocab_size}")

    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Pre-normalization and pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # -----------------------------------------------------------
    # 1. Define Special Tokens
    # -----------------------------------------------------------
    special_tokens = [
        "<|endoftext|>",  # 0
        "<|im_start|>",  # 1
        "<|im_end|>",  # 2
        "<think>",  # 3
        "</think>",  # 4
        "<|image_pad|>",  # 5
        "<tool_call>",  # 6
        "</tool_call>",  # 7
        "<tool_response>",  # 8
        "</tool_response>",  # 9
        "<tools>",  # 10
        "</tools>",  # 11
    ]

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Train
    logger.info(f"Training tokenizer on {files}...")
    tokenizer.train_from_iterator(
        batch_iterator(files, logger, max_bytes=max_train_bytes), trainer=trainer
    )

    # Simple assertion check
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2
    assert tokenizer.token_to_id("<think>") == 3

    # Post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Save the raw tokenizer
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))

    # Wrap in Transformers PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=special_tokens[
            1:
        ],  # All except bos/eos/unk/pad which is mapped to endoftext
    )

    # -----------------------------------------------------------
    # 2. Set Chat Template
    # -----------------------------------------------------------
    fast_tokenizer.chat_template = get_chat_template()

    fast_tokenizer.save_pretrained(save_path)
    logger.info(f"Tokenizer saved to {save_path} with Lulu support.")
    return fast_tokenizer


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description="Tokenizer Training Script")
    parser.add_argument("--data_path", type=str, default="data/pretrain_hq.jsonl")
    args = parser.parse_args()
    data_file = args.data_path

    # Check if the data file exists, otherwise create a dummy one
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Creating a dummy jsonl.")
        dummy_file = "dummy_data.jsonl"
        with open(dummy_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "Test data."}) + "\n")
        files_to_train = [dummy_file]
    else:
        files_to_train = [data_file]

    # Train
    ONE_GB = 1 * 1024 * 1024 * 1024
    tokenizer = train_tokenizer(files_to_train, vocab_size=6400, max_train_bytes=ONE_GB)

    # ==========================================
    # Test Chat Template
    # ==========================================
    print("\n=== Testing Chat Template ===")

    # Simulate a Tool Call conversation
    messages = [
        {"role": "user", "content": "What is the weather in Singapore?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "Singapore"},
                    }
                }
            ],
        },
        {"role": "tool", "content": '{"temp": 30, "condition": "Sunny"}'},
    ]

    # Note: To test the 'tools' part of the template, we need to pass tools to apply_chat_template
    # But apply_chat_template in older transformers might not support kwargs easily or we need to pass it in a specific way.
    # However, for basic structure check:

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print("Formatted String (No Tools passed):")
    print(formatted_prompt)

    # Verify special tokens
    assert "<tool_call>" in formatted_prompt
    assert "<tool_response>" in formatted_prompt
    assert "<|im_start|>" in formatted_prompt
