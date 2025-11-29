import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.dense import LuluConfig, LuluModel
# Import other models if needed


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.to(device)
    model.eval()

    # Format prompt with special tokens
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)

    print(f"Prompt: {prompt}")
    print("-" * 20)
    print("Generating...")

    # Simple generation loop
    # In a real scenario, use model.generate() from transformers
    # But since we implemented custom models, we might need to ensure they support it or write a simple loop
    # Our custom models inherit from PreTrainedModel, so they *should* support generate if configured correctly.
    # However, for educational transparency, let's write a simple greedy loop or use the built-in if it works.

    # Let's try using the built-in generate which is available on PreTrainedModel
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # Extract the assistant part
    # We look for the last <|im_start|>assistant
    try:
        response = generated_text.split("<|im_start|>assistant\n")[-1]
    except:
        response = generated_text

    # Handle Think Mode display
    if "<think>" in response and "</think>" in response:
        parts = response.split("</think>")
        thinking = parts[0].replace("<think>", "").strip()
        answer = parts[1].replace("<|im_end|>", "").strip()

        print("\n=== 🧠 Thinking Process ===")
        print(thinking)
        print("===========================\n")
        print("=== 💡 Final Answer ===")
        print(answer)
    else:
        print("\n=== Response ===")
        print(response.replace("<|im_end|>", "").strip())


if __name__ == "__main__":
    # Example usage with a dummy model
    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("lulu_tokenizer")
    except:
        print(
            "Tokenizer not found in lulu_tokenizer. Please run train_tokenizer.py first to generate it."
        )
        exit()

    # 2. Load Model (Dummy initialization for demo)
    config = LuluConfig(vocab_size=len(tokenizer), hidden_size=64, num_hidden_layers=2)
    model = LuluModel(config)

    # 3. Generate
    generate(model, tokenizer, "Solve 2 + 2")
