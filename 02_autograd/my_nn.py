import itertools
from collections.abc import Iterator
from typing import Any, Self

import numpy as np

from my_tensor import Parameter, Tensor, relu, tensor


class Module:
    _modules: dict[str, Self]
    _parameters: dict[str, Parameter]

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}

    def modules(self):
        """Return the direct child modules of this module."""
        return self.__dict__["_modules"].values()

    def parameters_dict(self):
        def get_parameters(module, prefix=""):
            for k, v in module._parameters.items():
                yield prefix + k, v
            for prefix, submodule in module._modules.items():
                yield from get_parameters(submodule, prefix=prefix + ".")

        return dict(get_parameters(self))

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Return an iterator over Module parameters.

        recurse: if True, the iterator includes parameters of submodules, recursively.
        """
        if not recurse:
            return self.__dict__["_parameters"].values()
        return itertools.chain(
            self.__dict__["_parameters"].values(),
            itertools.chain.from_iterable(
                m.parameters(recurse=True) for m in self.modules()
            ),
        )

    def __setattr__(self, key: str, val: Any) -> None:
        # If val is a Parameter or Module, store it in the appropriate _parameters or
        # _modules dict.
        # Otherwise, do the normal thing.
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Parameter | Self:
        # If key is in _parameters or _modules, return the corresponding value.
        # Otherwise, raise AttributeError.
        # only invoked if the attribute wasn't found the usual ways
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]
        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        raise AttributeError(f"{key} not in parameters or modules")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward!")

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return (
            f"Parameters={list(self.__dict__['_parameters'])}, "
            f"Modules={list(self.__dict__['_modules'])}, "
            f"extra={self.extra_repr()}"
        )


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True) -> None:
        """A simple linear (technically, affine) transformation."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_val = in_features**-0.5
        self.weight = Parameter(
            tensor(np.random.uniform(-init_val, init_val, (out_features, in_features)))
        )
        self.bias = (
            Parameter(tensor(np.random.uniform(-init_val, init_val, (out_features,))))
            if bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        _x_batches, x_in = x.shape
        _w_out, w_in = self.weight.shape
        assert x_in == w_in
        out = x @ self.weight.permute((1, 0))
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return f"{self.in_features=} {self.out_features=}"


class MLP(Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = Linear(28 * 28, 64)
        self.linear2 = Linear(64, 64)
        self.output = Linear(64, 10)

    def forward(self, x):
        x = x.reshape((x.shape[0], 28 * 28))
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        return self.output(x)


def cross_entropy(logits: Tensor, true_labels: Tensor) -> Tensor:
    """Like torch.nn.functional.cross_entropy with reduction='none'.

    logits: shape (batch, classes)
    true_labels: shape (batch,)

    Return: shape (batch, ) containing the per-example loss.
    """
    exp_logits = logits.exp()
    soft_argmax = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
    return -soft_argmax[
        (tensor(np.arange(len(logits)), requires_grad=False), true_labels)
    ].log()
