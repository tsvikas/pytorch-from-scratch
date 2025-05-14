import time
from pathlib import Path

import torch
import torch.utils.data
from torchvision import datasets, transforms
from tqdm.auto import tqdm

CACHE_DIR = Path(__file__).parents[3]
assert CACHE_DIR.joinpath("src").exists()
DATASET_CACHE = CACHE_DIR / "dataset_cache"


def get_mnist(subsample=10):
    """Return MNIST data using the provided Tensor class."""
    mnist_train = datasets.MNIST(DATASET_CACHE, train=True, download=True)
    mnist_test = datasets.MNIST(DATASET_CACHE, train=False)

    # Preprocessing
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.28,), (0.35,))]
    )
    train_indexes = range(0, len(mnist_train), subsample)
    train_reduced = [mnist_train[i] for i in train_indexes]
    train_tensors = torch.utils.data.TensorDataset(
        torch.stack(
            [transform(img) for img, label in tqdm(train_reduced, desc="Training data")]
        ),
        torch.tensor([label for img, label in train_reduced]),
    )

    test_indexes = range(0, len(mnist_test), subsample)
    test_reduced = [mnist_test[i] for i in test_indexes]
    test_tensors = torch.utils.data.TensorDataset(
        torch.stack(
            [transform(img) for img, label in tqdm(test_reduced, desc="Test data")]
        ),
        torch.tensor([label for img, label in test_reduced]),
    )

    train_loader = torch.utils.data.DataLoader(
        train_tensors, shuffle=True, batch_size=512
    )
    test_loader = torch.utils.data.DataLoader(test_tensors, batch_size=512)
    return train_loader, test_loader
