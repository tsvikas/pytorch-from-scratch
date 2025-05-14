from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import pooch
import torch
import torchvision
from PIL import Image
from tqdm.auto import tqdm

from . import labels

if TYPE_CHECKING:
    from collections.abc import Callable

CACHE_DIR = Path(__file__).parents[3]
assert CACHE_DIR.joinpath("src").exists()
DATASET_CACHE = CACHE_DIR / "dataset_cache"

IMAGE_URLS = {
    "https://anipassion.com/ow_userfiles/plugins/animal/breed_image_56efffab3e169.jpg": "1b65bb9f256ce36aa3d8e78a361925ff4b8ab84d1f97f8173cebe8e1c268de7b",
    "https://upload.wikimedia.org/wikipedia/commons/f/f2/Platypus.jpg": "01d8c764958cfb2441b3a240cda06451e50caf732061d34d4d90df603618d4dc",
    "https://static5.depositphotos.com/1017950/406/i/600/depositphotos_4061551-stock-photo-hourglass.jpg": "359e4696b26cb75390376473c4ae42b3f41c592b72884b1fc6c22060664251ac",
    "https://img.nealis.fr/ptv/img/p/g/1465/1464424.jpg": "b35d387bae6ad69a7bbd09d5e023f19dd3d8be3ab8e546c1dd18a6b6e1dad3de",
    "https://ychef.files.bbci.co.uk/976x549/p0639ffn.jpg": "782b225b9587bd07ffec6ae7abc373717862961d3d21c5a76cef4374a33b69c8",
    "https://www.thoughtco.com/thmb/Dk3bE4x1qKqrF6LBf2qzZM__LXE=/1333x1000/smart/filters:no_upscale()/iguana2-b554e81fc1834989a715b69d1eb18695.jpg": "08950d9dc887a56f0f6951de3fe3afac5a64a2f60ba46c14ac5016266431adc9",
    "https://i.redd.it/mbc00vg3kdr61.jpg": "0d4d23317dc1051ff68dac8dbce40134ce73f2bc3a81bf9d14322f7fdeadf777",
    "https://static.wikia.nocookie.net/disneyfanon/images/a/af/Goofy_pulling_his_ears.jpg": "493e174d9d7cf9b73988779e58e8b39bfbfdde056898059928bd654fb329df3c",
}
IMAGENET_LABELS = labels.imagenet
CIFAR_LABELS = labels.cifar


def load_image(url: str, known_hash: str | None = None) -> Image.Image:
    """Return the image at the specified URL, using a local cache if possible."""
    filename = pooch.retrieve(url, known_hash=known_hash)
    return Image.open(filename)

def load_all_images() -> list[Image.Image]:
    return [load_image(url, known_hash) for url, known_hash in tqdm(IMAGE_URLS.items())]

def prepare_data(images: list[Image.Image]) -> torch.Tensor:
    """Preprocess each image and stack them into a single tensor.

    Return: shape (batch=len(images), num_channels=3, height=224, width=224)
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    preprocess: Callable[[Image.Image], torch.Tensor] = torchvision.transforms.Compose(
        [
            # H W C -> C H W
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(mean, std),
        ]
    )  # equivalent to: models.ResNet34_Weights.DEFAULT.transforms()
    return torch.stack([preprocess(img) for img in tqdm(images)])


def get_cifar10() -> tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """Download (if necessary) and return the CIFAR10 dataset."""
    # Magic constants taken from: https://docs.ffcv.io/ffcv_examples/cifar10.html
    mean = torch.tensor([125.307, 122.961, 113.8575]) / 255
    std = torch.tensor([51.5865, 50.847, 51.255]) / 255
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)]
    )
    cifar_train = torchvision.datasets.CIFAR10(
        DATASET_CACHE, transform=transform, download=True, train=True
    )
    cifar_test = torchvision.datasets.CIFAR10(
        DATASET_CACHE, transform=transform, download=True, train=False
    )
    return cifar_train, cifar_test
