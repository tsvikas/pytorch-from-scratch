import time

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pytorch_from_scratch.p04_BERT.get_reviews import get_reviews_datasets
from pytorch_from_scratch.p04_BERT.my_bert import (
    Bert,
    BertConfig,
    BertOutput,
    load_pretrained_weights,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertClassifier(nn.Module):
    def __init__(self, config: BertConfig):
        self.bert = Bert(config)
        self.dropout = nn.Dropout(config.dropout)
        self.class_sentiment = nn.Linear(config.vocab_size, 2)
        self.class_stars = nn.Linear(config.vocab_size, 1)

    def forward(self, x: torch.Tensor) -> BertOutput:
        logits = x = self.bert(x)
        x = self.dropout(x)
        sentiment = self.class_sentiment(x)
        stars = torch.clip(self.class_stars(x) * 5 + 5, 1, 10)
        return BertOutput(logits=logits, is_positive=sentiment, star_rating=stars)


def prepare_bert():
    config_dict = dict(lr=2e-05, batch_size=8, step_every=2, epochs=2)
    wandb.init(project="w2d2_imdb", config=config_dict)
    config = wandb.config
    model = BertClassifier(BertConfig())
    model.bert = load_pretrained_weights()
    for p in model.parameters():
        p.requires_grad_(True)
    return model, config


def finetune_bert(model, config, train_data):
    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True
    )
    model.train().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
    optimizer.zero_grad()
    classification_loss_fn = torch.nn.CrossEntropyLoss()
    wandb.watch(model, criterion=classification_loss_fn, log="all", log_freq=100)
    examples_seen = 0
    start_time = time.time()
    for _epoch in range(config.epochs):
        for i, (input_ids, y_positive, _y_stars) in enumerate(tqdm(train_loader)):
            input_ids = input_ids.to(device)
            y_positive = y_positive.long().to(device)
            out = model(input_ids)
            if torch.isnan(out.is_positive).any():
                raise ValueError("NaN detected!")
            loss = classification_loss_fn(out.is_positive, y_positive)
            loss.backward()
            if i > 0 and i % config.step_every == 0:
                optimizer.step()
                optimizer.zero_grad()
            examples_seen += len(input_ids)
            if i % 20 == 0:
                wandb.log(
                    dict(
                        train_loss=loss,
                        elapsed=time.time() - start_time,
                        step=examples_seen,
                    )
                )
    return model, examples_seen


def evaluate(model, test_data, step, batch_size):
    test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)
    test_loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        n_correct = 0
        n_total = 0
        loss_total = 0.0
        for i, (input_ids, y_positive, _y_stars) in enumerate(tqdm(test_loader)):
            input_ids = input_ids.to(device)
            y_positive = y_positive.long().to(device)
            out = model(input_ids)
            loss_total += test_loss_fn(out.is_positive, y_positive).item()
            n_correct += (out.is_positive.argmax(dim=-1) == y_positive).sum().item()
            n_total += len(y_positive)
            if i == 50:
                break
        wandb.log(
            dict(
                test_loss=loss_total / n_total,
                test_accuracy=n_correct / n_total,
                step=step,
            )
        )


def main():
    train_data, test_data = get_reviews_datasets()
    model, config = prepare_bert()
    model, steps = finetune_bert(model, config, train_data)
    evaluate(model, test_data, steps, config.batch_size)


if __name__ == "__main__":
    main()
