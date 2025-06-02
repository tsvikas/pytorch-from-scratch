import torch

from pytorch_from_scratch.p04_BERT.my_bert import (
    Bert,
    load_pretrained_weights,
    load_tokenizer,
)


def predict(model: Bert, tokenizer, text: str, k=15) -> list[list[str]]:
    """
    Return a list of k strings for each [MASK] in the input.
    """
    tokens = tokenizer(text)
    model.eval()
    tokens_input = torch.tensor(tokens.input_ids).unsqueeze(0)
    tokens_output: torch.Tensor = model(tokens_input).logits
    tokens_output_masked = tokens_output[tokens_input == tokenizer.mask_token_id]
    pred = tokens_output_masked.topk(k).indices
    pred_tokens = [[tokenizer.decode(pp) for pp in p] for p in pred.tolist()]
    return pred_tokens


def test_bert_prediction(predict, model, tokenizer):
    """Your Bert should know some names of American presidents."""
    text = "Former President of the United States of America, George[MASK][MASK]"
    predictions = predict(model, tokenizer, text)
    assert "Washington" in predictions[0]
    assert "Bush" in predictions[0]


def run_model():
    my_bert = load_pretrained_weights()
    tokenizer = load_tokenizer()
    test_bert_prediction(predict, my_bert, tokenizer)
    your_text = "The Answer to the Ultimate Question of Life, The Universe, and Everything is [MASK]."
    predictions = predict(my_bert, tokenizer, your_text)
    print("Model predicted: \n", "\n".join(map(str, predictions)))


if __name__ == "__main__":
    run_model()
