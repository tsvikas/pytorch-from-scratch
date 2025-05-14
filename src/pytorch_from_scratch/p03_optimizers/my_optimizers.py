from collections.abc import Iterable

import torch


class SGD:
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float,
        momentum: float,
        weight_decay: float,
    ):
        """Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

        """
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.b = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @torch.inference_mode()
    def step(self) -> None:
        for i, p in enumerate(self.params):
            assert p.grad is not None
            g = p.grad + self.weight_decay * p
            if self.momentum:
                self.b[i] = self.momentum * self.b[i] + g
                g = self.b[i]
            p -= self.lr * g


class RMSprop:
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float,
        alpha: float,
        eps: float,
        weight_decay: float,
        momentum: float,
    ):
        """Implement RMSprop.

        Like the PyTorch version, but assume centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop

        """
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.v = [torch.zeros_like(p) for p in self.params]
        self.b = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @torch.inference_mode()
    def step(self) -> None:
        for i, p in enumerate(self.params):
            assert p.grad is not None
            g = p.grad + self.weight_decay * p
            v = self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * g**2
            if self.momentum > 0:
                self.b[i] = self.momentum * self.b[i] + g / (v.sqrt() + self.eps)
                p -= self.lr * self.b[i]
            else:
                p -= self.lr * g / (v.sqrt() + self.eps)


class Adam:
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        """Implement Adam.

        Like the PyTorch version, but assume amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        """
        self.params = list(params)
        self.lr = lr
        self.b1 = betas[0]
        self.b2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @torch.inference_mode()
    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            assert p.grad is not None
            g = p.grad + self.weight_decay * p
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.m[i] + (1 - self.b2) * g**2
            m_hat = self.m[i] / (1 - self.b1**self.t)
            v_hat = self.v[i] / (1 - self.b2**self.t)
            p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
