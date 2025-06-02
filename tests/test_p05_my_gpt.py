import torch
import transformers

from pytorch_from_scratch.p05_gpt2 import my_gpt


def test_load_pretrained_weights():
    model = my_gpt.load_pretrained_weights()
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    def encode(text: str) -> torch.Tensor:
        """Return a Tensor of shape (batch=1, seq)."""
        return tokenizer(text, return_tensors="pt")["input_ids"]

    prompt = "Former President of the United States of America, George"
    input_ids = encode(prompt)
    logits = model(input_ids)[0, -1]
    topk = torch.topk(logits, k=10).indices
    next_tokens = tokenizer.batch_decode(topk.reshape(-1, 1))
    assert " Washington" in next_tokens
    assert " Bush" in next_tokens


def test_sample_zero_temperature():
    model = my_gpt.load_pretrained_weights()
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    prompt = "Jingle bells, jingle bells, jingle all the way"
    output = my_gpt.sample_tokens(
        model, tokenizer, prompt, temperature=0, max_tokens_generated=8
    )
    expected = (
        "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
    )
    assert output == expected
