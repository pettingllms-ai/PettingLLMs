from transformers import AutoTokenizer

# 你要测试的模型列表
model_names = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-Coder-3B"
]

# 一个测试的对话
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "请帮我写一首诗。"},
    {"role": "assistant", "content": "好的，下面是代码示例..."}
]

def test_models(model_names, messages):
    for name in model_names:
        print(f"\n===== Testing {name} =====")
        tokenizer = AutoTokenizer.from_pretrained(name)

        # enable_thinking = True
        text_thinking = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True
        )
        print("[enable_thinking=True]")
        print(text_thinking)

        # enable_thinking = False
        text_no_thinking = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        )
        print("\n[enable_thinking=False]")
        print(text_no_thinking)

if __name__ == "__main__":
    test_models(model_names, messages)
