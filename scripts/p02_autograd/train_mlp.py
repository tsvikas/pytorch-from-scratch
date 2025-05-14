import time
from collections.abc import Iterable

from pytorch_from_scratch.p02_autograd import my_tensor
from pytorch_from_scratch.p02_autograd.get_images import get_mnist
from pytorch_from_scratch.p02_autograd.my_nn import MLP, Module, cross_entropy
from pytorch_from_scratch.p02_autograd.my_tensor import NoGrad, Parameter, Tensor

my_tensor.grad_tracking_enabled = True


class SGD:
    def __init__(self, params: Iterable[Parameter], lr: float) -> None:
        """Vanilla SGD with no additional features."""
        self.params = list(params)
        self.lr = lr
        self.b = [None for _ in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        with NoGrad():
            for _i, p in enumerate(self.params):
                assert isinstance(p.grad, Tensor)
                p.add_(p.grad, -self.lr)


def train(model: Module, train_loader, optimizer: SGD):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = Tensor(data.numpy())
        target = Tensor(target.numpy())
        optimizer.zero_grad()
        output = model(data)
        loss = cross_entropy(output, target).sum() / len(output)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0 or (batch_idx + 1 == len(train_loader)):
            pass
        if batch_idx % 50 == 0 or (batch_idx + 1 == len(train_loader)):
            print(
                f"Train: "
                f"[{(batch_idx + 1) * len(data)}/{len(train_loader.dataset)} "
                f"({(batch_idx + 1) / len(train_loader):.0%})]\t"
                f"Loss: {loss.item():.6f}"
            )
    return loss


def test(model: Module, test_loader) -> None:
    test_loss = 0
    correct = 0
    with NoGrad():
        for data, target in test_loader:
            data = Tensor(data.numpy())
            target = Tensor(target.numpy())
            output = model(data)
            test_loss += cross_entropy(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += (pred == target.reshape(pred.shape)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        "Test: "
        f"Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({correct / len(test_loader.dataset):.0%})\n"
    )


def main() -> None:
    train_loader, test_loader = get_mnist()

    num_epochs = 50
    model = MLP()
    start = time.time()
    optimizer = SGD(model.parameters(), 0.01)
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}/{num_epochs}")
        train(model, train_loader, optimizer)
        test(model, test_loader)
        optimizer.step()
    print(f"Completed in {time.time() - start: .2f}s")


if __name__ == "__main__":
    main()
