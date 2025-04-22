import json
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import requests
import torch
import torchvision
from PIL import Image
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

IMAGE_URLS = [
    "https://anipassion.com/ow_userfiles/plugins/animal/breed_image_56efffab3e169.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/f2/Platypus.jpg",
    "https://static5.depositphotos.com/1017950/406/i/600/depositphotos_4061551-stock-photo-hourglass.jpg",
    "https://img.nealis.fr/ptv/img/p/g/1465/1464424.jpg",
    "https://ychef.files.bbci.co.uk/976x549/p0639ffn.jpg",
    "https://www.thoughtco.com/thmb/Dk3bE4x1qKqrF6LBf2qzZM__LXE=/1333x1000/smart/filters:no_upscale()/iguana2-b554e81fc1834989a715b69d1eb18695.jpg",
    "https://i.redd.it/mbc00vg3kdr61.jpg",
    "https://static.wikia.nocookie.net/disneyfanon/images/a/af/Goofy_pulling_his_ears.jpg",
]
IMAGE_CACHE = Path("image_cache")
IMAGENET_LABELS = {
    int(k): label
    for k, label in json.loads(Path("imagenet_labels.json").read_text()).items()
}
CIFAR_LABELS = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def load_image(url: str) -> Image.Image:
    """Return the image at the specified URL, using a local cache if possible.

    For a robust implementation, use pooch.retrieve
    """
    IMAGE_CACHE.mkdir(exist_ok=True)
    filename = IMAGE_CACHE / url.rsplit("/", 1)[1].replace("%20", "")
    if filename.exists():
        data = filename.read_bytes()
    else:
        response = requests.get(url)
        data = response.content
        filename.write_bytes(data)
    assert data, f"{filename}"
    return Image.open(BytesIO(data))


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


def get_cifar10(
    cache_dir: Path = Path("CIFAR10"),
) -> tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """Download (if necessary) and return the CIFAR10 dataset."""
    # Magic constants taken from: https://docs.ffcv.io/ffcv_examples/cifar10.html
    mean = torch.tensor([125.307, 122.961, 113.8575]) / 255
    std = torch.tensor([51.5865, 50.847, 51.255]) / 255
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)]
    )
    cifar_train = torchvision.datasets.CIFAR10(
        cache_dir, transform=transform, download=True, train=True
    )
    cifar_test = torchvision.datasets.CIFAR10(
        cache_dir, transform=transform, download=True, train=False
    )
    return cifar_train, cifar_test
