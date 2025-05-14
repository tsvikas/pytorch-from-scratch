import json
from importlib import resources

files = resources.files(__name__)
imagenet = {
    int(k): label
    for k, label in json.loads(
        files.joinpath("imagenet_labels.json").read_text()
    ).items()
}
cifar = {
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

__all__ = ["imagenet", "cifar"]
