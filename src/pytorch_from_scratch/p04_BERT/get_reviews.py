import tarfile
from dataclasses import dataclass
from pathlib import Path

import pooch
import torch

from pytorch_from_scratch.p04_BERT.my_bert import load_tokenizer

from ..utils import TensorDataset

CACHE_DIR = Path(__file__).parents[3]
assert CACHE_DIR.joinpath("src").exists()
DATASET_CACHE = CACHE_DIR / "dataset_cache"


@dataclass(frozen=True)
class Review:
    split: str
    is_positive: bool
    stars: int
    text: str


def get_reviews() -> list[Review]:
    path = pooch.retrieve(
        url="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        known_hash="md5:7c2ac02c03563afcf9b574c7e56c153a",
    )
    reviews = []
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            if not (member.isfile() and member.path.count("/") == 3):
                continue
            base, dataset, sentiment, id_stars = member.path.split("/")
            assert base == "aclImdb"
            if dataset not in ["test", "train"]:
                continue
            if sentiment == "unsup":
                continue
            assert sentiment in ["pos", "neg"]
            assert id_stars.endswith(".txt")
            _review_id, stars = id_stars.removesuffix(".txt").split("_")
            reviews.append(
                Review(
                    dataset,
                    sentiment != "neg",
                    int(stars),
                    tar.extractfile(member).read().decode("utf8"),
                )
            )
    assert sum(r.split == "train" for r in reviews) == 25000
    assert sum(r.split == "test" for r in reviews) == 25000
    return reviews


def reviews_to_dataset(reviews: list[Review], tokenizer) -> TensorDataset:
    """Tokenize the reviews and bundle into a TensorDataset.

    The tensors in the dataset should be:

    input_ids: shape (batch, sequence length), dtype int
    labels: shape (batch, ), dtype int
    """
    input_ids = torch.tensor(
        tokenizer(
            [review.text for review in reviews], padding="max_length", truncation=True
        )["input_ids"]
    )
    labels = torch.tensor([review.stars for review in reviews])
    return TensorDataset(input_ids, labels)


def get_reviews_datasets(saved_tokens_path=DATASET_CACHE / "imdb_reviews_tokens.pt"):
    reviews = get_reviews()
    tokenizer = load_tokenizer()
    if saved_tokens_path.exists():
        train_data, test_data = torch.load(saved_tokens_path)
    else:
        train_data = reviews_to_dataset(
            [r for r in reviews if r.split == "train"], tokenizer
        )
        test_data = reviews_to_dataset(
            [r for r in reviews if r.split == "test"], tokenizer
        )
        torch.save((train_data, test_data), saved_tokens_path)
    return train_data, test_data
