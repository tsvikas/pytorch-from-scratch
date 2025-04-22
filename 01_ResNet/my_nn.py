import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter

IntOrPair = int | tuple[int, int]
Pair = tuple[int, int]


def pad2d(
    x: Tensor, left: int, right: int, top: int, bottom: int, pad_value: float
) -> Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    """
    padded = x.new_full(
        size=[*x.size()[:-2], top + bottom + x.size()[-2], left + right + x.size()[-1]],
        fill_value=pad_value,
    )
    padded[..., top : -bottom or None, left : -right or None] = x
    return padded


def force_pair(v: IntOrPair) -> Pair:
    """Convert v to a pair of int, if it isn't already."""
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    if isinstance(v, int):
        return (v, v)
    raise ValueError(v)


def conv2d(x: Tensor, weights: Tensor, stride: IntOrPair = 1, padding: IntOrPair = 0):
    """Like torch's conv2d using bias=False.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    """
    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)
    x = pad2d(x, padding_w, padding_w, padding_h, padding_h, 0)
    batch, in_channels, height, width = x.size()
    _out_channels, in_channels0, kernel_height, kernel_width = weights.size()
    assert in_channels == in_channels0

    *x_strides, in_stride_h, in_stride_w = tuple(x.stride())
    sliding_input = x.as_strided(
        size=(
            batch,
            in_channels,
            (height - kernel_height) // stride_h + 1,
            kernel_height,
            (width - kernel_width) // stride_w + 1,
            kernel_width,
        ),
        stride=(
            *x_strides,
            in_stride_h * stride_h,
            in_stride_h,
            in_stride_w * stride_w,
            in_stride_w,
        ),
    )
    return torch.einsum("bixkyj, oikj -> boxy", sliding_input, weights)


def maxpool2d(
    x: Tensor,
    kernel_size: IntOrPair,
    stride: IntOrPair | None = None,
    padding: IntOrPair = 0,
) -> Tensor:
    """Like torch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, height, width)
    """
    if stride is None:
        stride = kernel_size
    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)
    x = pad2d(x, padding_w, padding_w, padding_h, padding_h, float("-inf"))
    batch, in_channels, height, width = x.size()
    kernel_h, kernel_w = force_pair(kernel_size)
    in_stride = list(x.stride())
    in_stride_h, in_stride_w = in_stride[-2:]
    sliding_input = x.as_strided(
        size=(
            batch,
            in_channels,
            (height - kernel_h) // stride_h + 1,
            kernel_h,
            (width - kernel_w) // stride_w + 1,
            kernel_w,
        ),
        stride=tuple(
            in_stride[:-2]
            + [in_stride_h * stride_h, in_stride_h, in_stride_w * stride_w, in_stride_w]
        ),
    )
    return sliding_input.amax(dim=[-1, -3])  # bixkyj -> bixy


def extra_attr(obj, attrs):
    return ", ".join(f"{attr}={getattr(obj, attr)}" for attr in attrs)


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: IntOrPair,
        stride: IntOrPair | None = None,
        padding: IntOrPair = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return maxpool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        return extra_attr(self, ["kernel_size", "stride", "padding"])


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True) -> None:
        """A simple linear (technically, affine) transformation."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_val = in_features**-0.5
        self.weight = Parameter(
            torch.empty((out_features, in_features)).uniform_(-init_val, init_val)
        )
        self.bias = (
            Parameter(torch.empty((out_features,)).uniform_(-init_val, init_val))
            if bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        mul = torch.einsum("...i,oi->...o", x, self.weight)
        if self.bias is not None:
            mul += self.bias
        return mul

    def extra_repr(self) -> str:
        return extra_attr(self, ["in_features", "out_features"])


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
    ) -> None:
        """Same as torch.nn.Conv2d with bias=False."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kernel_height, kernel_width = force_pair(kernel_size)
        weight_size = out_channels, in_channels, kernel_height, kernel_width
        init_val = (in_channels * kernel_height * kernel_width) ** -0.5
        self.weight = Parameter(torch.empty(weight_size).uniform_(-init_val, init_val))

        self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply the functional conv2d."""
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        return extra_attr(
            self, ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        )


class BatchNorm2d(Module):
    running_mean: Tensor
    "running_mean: shape (num_features,)"
    running_var: Tensor
    "running_var: shape (num_features,)"
    num_batches_tracked: Tensor
    "num_batches_tracked: shape ()"

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1) -> None:
        """Like nn.BatchNorm2d with track_running_stats=True, affine=True."""
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # learnable parameters, shape (num_features,)
        self.weight = Parameter(torch.ones((num_features,)))
        self.bias = Parameter(torch.zeros((num_features,)))

        self.register_buffer("running_mean", torch.zeros((num_features,)))
        self.register_buffer("running_var", torch.ones((num_features,)))
        self.register_buffer("num_batches_tracked", torch.tensor(0))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize each channel.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """
        if self.training:
            # use the batch mean and variance
            mean = x.mean(dim=[0, 2, 3])
            var = torch.var(x, dim=[0, 2, 3], unbiased=False)
            # update the running mean and variance.
            m = self.momentum
            # self.running_mean = (1 - m) * self.running_mean + m * mean
            self.running_mean *= 1 - m
            self.running_mean += m * mean
            # self.running_var = (1 - m) * self.running_var + m * var
            self.running_var *= 1 - m
            self.running_var += m * var
            self.num_batches_tracked += 1
        else:
            # use the running mean and variance.
            mean = self.running_mean
            var = self.running_var
        mean = mean.reshape((-1, 1, 1))
        var = var.reshape((-1, 1, 1))
        weight = self.weight.reshape((-1, 1, 1))
        bias = self.bias.reshape((-1, 1, 1))
        normed = (x - mean) / (var + self.eps) ** 0.5 * weight + bias
        return normed

    def extra_repr(self) -> str:
        return extra_attr(self, ["num_features", "eps", "momentum"])


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.maximum(x, torch.tensor(0.0))


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        """Call `self.add_module` on each provided module, giving each one a unique name.

        Internally, this adds them to the dictionary `self._modules` in the base class,
        which means they'll be included in self.parameters() as desired.
        """
        super().__init__()
        for i, module in enumerate(modules):
            name = f"{i:02}_{module.__class__.__name__}"
            self.add_module(name, module)

    def forward(self, x: Tensor) -> Tensor:
        """Chain modules."""
        for module in self._modules.values():
            assert module is not None
            x = module(x)
        return x


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """Flatten out dimensions from start_dim to end_dim, inclusive of both.

        Return a view if possible, otherwise a copy.
        """
        start = self.start_dim
        end = self.end_dim + 1 or input.dim()
        left, mid, right = (
            input.shape[:start],
            input.shape[start:end],
            input.shape[end:],
        )
        prod = 1
        for s in mid:
            prod *= s
        new_shape = (*left, prod, *right)
        return input.reshape(new_shape)

    def extra_repr(self) -> str:
        return extra_attr(self, ["start_dim", "end_dim"])


class AveragePool(Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        return x.mean(dim=[-2, -1], keepdim=True)
