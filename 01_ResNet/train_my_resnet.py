import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import get_images
import my_resnet

MODEL_FILENAME = Path("model_save/resnet34_cifar10.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512


def train(model, train_loader: DataLoader, epochs: int) -> None:
    """Simple train loop."""
    print(f"{torch.cuda.get_device_name(DEVICE) = }")
    print(f"{torch.cuda.memory_allocated(DEVICE) = } bytes")
    model.to(DEVICE).train()
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print(f"[{time.time() - start_time: >6.2f}] {epoch = }")
        for _i, (x, y) in enumerate(tqdm(train_loader, leave=False)):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
        torch.save(model, MODEL_FILENAME)


def main() -> None:
    assert torch.cuda.is_available()
    cifar_train, _cifar_test = get_images.get_cifar10()
    train_loader = DataLoader(
        cifar_train, batch_size=BATCH_SIZE, pin_memory=False, shuffle=True
    )
    model = my_resnet.ResNet34(n_classes=10)
    train(model, train_loader, epochs=8)


if __name__ == "__main__":
    main()
