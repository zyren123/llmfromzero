from transformers import AutoTokenizer


def test_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("lulu_tokenizer")
    print(tokenizer.tokenize("Hello, world!"))
    print(tokenizer.decode(tokenizer.encode("Hello, world!")))
    print(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello, world!"}], tokenize=False
        )
    )
    print(tokenizer.encode("<|im_start|>你好呀，今天的天气怎么样？"))
    print(tokenizer.decode(tokenizer.encode("<|im_start|>你好呀，今天的天气怎么样？")))


if __name__ == "__main__":
    test_tokenizer()
