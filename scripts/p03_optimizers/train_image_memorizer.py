from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageMemorizer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.dims = [in_dim, hidden_dim, out_dim]
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)


class TensorDataset:
    def __init__(self, *tensors: torch.Tensor):
        """Validate the sizes and store the tensors in a field named `tensors`."""
        if tensors:
            shape = tensors[0].shape
            for tensor in tensors:
                if tensor.shape[0] != shape[0]:
                    raise ValueError(
                        "all tensors should have the same length in the first domension"
                    )
        self.tensors = tensors

    def __getitem__(self, index: int | slice) -> tuple[torch.Tensor, ...]:
        """Return a tuple of length len(self.tensors) with the index applied to each."""
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        """Return the size in the first dimension, common to all the tensors."""
        return self.tensors[0].shape[0]


def all_coordinates_scaled(height: int, width: int) -> torch.Tensor:
    """Return a tensor of shape (height*width, 2) where each row is a (x, y) coordinate.

    The range of x and y should be from [-1, 1] in both height and width dimensions.
    """
    coords = torch.tensor([(x, y) for y in range(height) for x in range(width)])
    coords = coords - coords.amin(dim=0)
    coords = coords / coords.amax(dim=0)
    coords = coords * 2 - 1
    return coords


def preprocess_image(img: Image.Image) -> TensorDataset:
    """Convert an image into a supervised learning problem predicting (R, G, B) given (x, y).

    Return: TensorDataset wrapping input and label tensors.
    input: shape (num_pixels, 2)
    label: shape (num_pixels, 3)
    """
    img_tensor = transforms.ToTensor()(img)[:3]
    _channels, height, width = img_tensor.shape
    img_tensor = rearrange(img_tensor, "c h w -> (h w) c")  # type: torch.Tensor
    img_tensor *= 2
    img_tensor -= 1
    ys = (
        repeat(torch.arange(0, height), "h -> (h w)", h=height, w=width) / height * 2
        - 1
    )
    xs = repeat(torch.arange(0, width), "w -> (h w)", h=height, w=width) / width * 2 - 1
    coords = torch.stack([xs, ys], dim=-1)
    return TensorDataset(coords, img_tensor)


def train_test_split(
    all_data: TensorDataset, train_frac=0.8, val_frac=0.01, test_frac=0.01
) -> list[TensorDataset]:
    """Return [train, val, test] datasets containing the specified fraction of examples.

    If the fractions add up to less than 1, some of the data is not used.
    """
    perm = torch.randperm(len(all_data))
    train_size = int(train_frac * len(all_data))
    train_val_size = int((train_frac + val_frac) * len(all_data))
    train_val_test_size = int((train_frac + val_frac + test_frac) * len(all_data))
    train_perm = perm[0:train_size]
    val_perm = perm[train_size:train_val_size]
    test_perm = perm[train_val_size:train_val_test_size]
    return [
        TensorDataset(*all_data[train_perm]),
        TensorDataset(*all_data[val_perm]),
        TensorDataset(*all_data[test_perm]),
    ]


def to_grid(X: torch.Tensor, Y: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert preprocessed data from the format used in the DataSet back to an image tensor.

    X: shape (n_pixels, dim=2)
    Y: shape (n_pixels, channel=3)

    Return: shape (height, width, channels=3)
    """
    grid = torch.zeros(height, width, 3)
    y_coords = ((X[:, 1] + 1) / 2 * (height - 1) + 0.5).long()
    x_coords = ((X[:, 0] + 1) / 2 * (width - 1) + 0.5).long()
    grid[y_coords, x_coords] = (Y + 1) / 2
    return grid


def train_one_epoch(
    model: ImageMemorizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer = None,
) -> float:
    """Show each example in the model to the model once.

    Use `torch.optim.Adam` for the optimizer (you'll build your own Adam later today).
    Use `F.l1_loss(prediction, actual)` for the loss function. This just puts
    less weight on very bright or dark pixels, which seems to produce nicer images.

    Return: the total loss divided by the number of examples.
    """
    model.to(device)
    model.train()
    # it is probably NOT good to recreate the optimizer for each epoch, since it
    # losses its history
    optimizer = optimizer or torch.optim.Adam(model.parameters(recurse=True))
    total_loss = 0.0
    n_elems = 0
    with tqdm(total=len(dataloader.dataset)) as pbar:
        for xy, img in dataloader:
            optimizer.zero_grad()
            xy = xy.to(device)
            img = img.to(device)
            img_pred = model(xy)
            loss = F.l1_loss(img_pred, img)
            loss.backward()
            optimizer.step()
            n_elems += len(xy)
            total_loss += loss.item() * len(xy)
            pbar.update(len(xy))
    return total_loss / n_elems


def evaluate(model: ImageMemorizer, dataloader: DataLoader) -> float:
    """Return the total L1 loss over the provided data divided by the number of examples."""
    model.to(device)
    model.eval()  # Does nothing on this particular model, but good practice to have it
    total_loss = 0.0
    n_elems = 0
    with tqdm(total=len(dataloader.dataset)) as pbar:
        for xy, img in dataloader:
            xy = xy.to(device)
            img = img.to(device)
            with torch.inference_mode():
                img_pred = model(xy)
            loss = F.l1_loss(img_pred, img)
            n_elems += len(xy)
            total_loss += loss.item() * len(xy)
            pbar.update(len(xy))
    return total_loss / n_elems


def main():
    # metadata
    print(f"{device=}")

    # load input image
    filename = Path() / "vangogh.jpg"
    img = Image.open(filename)
    print(
        f"Image size in pixels: {img.size[0]} x {img.size[1]} = {img.size[0] * img.size[1]}"
    )
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"input image\n{img.size=}")

    # preprocess data
    all_data = preprocess_image(img)
    train_data, val_data, test_data = train_test_split(all_data)
    print(f"{len(all_data)=}, {len(train_data)=}, {len(val_data)=}, {len(test_data)=}")

    width, height = img.size
    X, Y = train_data.tensors
    fig, ax = plt.subplots()
    ax.imshow(to_grid(X, Y, width, height))
    ax.set_title("training data")
    plt.show()

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256)
    test_loader = DataLoader(test_data, batch_size=256)

    # setup model
    model = ImageMemorizer(in_dim=2, out_dim=3, hidden_dim=400)
    train_losses = []
    val_losses = []

    # training loop
    for epoch in range(30):
        optimizer = torch.optim.Adam(model.parameters(recurse=True))
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = evaluate(model, val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"\n{epoch=}, {train_loss=}, {val_loss=}", flush=True)
        if (epoch + 1) % 6 == 0:
            fig, ax = plt.subplots()
            ax.plot(range(len(train_losses)), train_losses, label="train")
            ax.plot(range(len(val_losses)), val_losses, label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("l1_loss")
            ax.legend()
            ax.set_title("training loss")
            plt.show()

    # create the memorized image
    xy = all_coordinates_scaled(img.height, img.width).to(device)
    with torch.inference_mode():
        img_pred = model(xy).clip(-1, 1).cpu()

    fig, ax = plt.subplots()
    ax.imshow(to_grid(xy, img_pred, img.width, img.height))
    ax.set_title("prediction")
    plt.show()


if __name__ == "__main__":
    main()
