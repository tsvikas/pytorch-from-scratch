import torch
import transformers

from pytorch_from_scratch.p05_gpt2 import my_gpt


def main():
    model = my_gpt.load_pretrained_weights()
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    token = my_gpt.sample_next_token(
        model,
        torch.tensor([10, 20, 30]),
        temperature=0.0,
        freq_penalty=2.0,
        cache=None,
    )
    print(f"next token for [10, 20, 30] = {token}")

    tokens = my_gpt.sample_tokens(
        model,
        tokenizer,
        "United States of",
        max_tokens_generated=3,
        temperature=0.5,
    )
    print(tokens)

    tokens = my_gpt.sample_tokens(
        model,
        tokenizer,
        "My life motto: ",
        max_tokens_generated=30,
        temperature=1.0,
        freq_penalty=2.0,
    )
    print(tokens)


if __name__ == "__main__":
    main()
