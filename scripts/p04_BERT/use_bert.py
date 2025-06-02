import torch
import transformers

from pytorch_from_scratch.p04_BERT.my_bert import Bert, BertConfig


def load_pretrained_weights(config: BertConfig) -> Bert:
    bert = Bert(config)
    hf_bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
    their_params = hf_bert.state_dict().copy()
    their_params.pop("cls.predictions.decoder.weight")
    their_params.pop("cls.predictions.bias")
    weights_to_load = {}
    assert len(their_params) == len(bert.state_dict())
    for loaded_key, my_key in zip(their_params, bert.state_dict()):
        weights_to_load[my_key] = their_params[loaded_key]
    bert.load_state_dict(weights_to_load)
    return bert


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
    my_bert = load_pretrained_weights(BertConfig())
    assert all(p.is_leaf for _name, p in my_bert.named_parameters())
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    test_bert_prediction(predict, my_bert, tokenizer)
    your_text = "The Answer to the Ultimate Question of Life, The Universe, and Everything is [MASK]."
    predictions = predict(my_bert, tokenizer, your_text)
    print("Model predicted: \n", "\n".join(map(str, predictions)))


if __name__ == "__main__":
    run_model()
