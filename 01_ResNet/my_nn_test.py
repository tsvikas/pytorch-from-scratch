import numpy as np
import torch
from pytest import mark
from torch.testing import assert_close

import my_nn


@mark.parametrize("conv2d", [my_nn.conv2d])
def test_conv2d_minimal(conv2d, n_tests=10) -> None:

    for _i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        w = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))

        x = torch.randn((b, ci, h, w))
        weights = torch.randn((co, ci, *kernel_size))
        my_output = conv2d(x, weights)
        torch_output = torch.conv2d(x, weights)
        assert_close(my_output, torch_output, atol=1e-4, rtol=1e-4)


@mark.parametrize("pad", [my_nn.pad2d])
def test_pad2d(pad) -> None:
    x = torch.arange(4).float().view((1, 1, 2, 2))
    expected = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        ]
    )
    actual = pad(x, 0, 1, 2, 3, 0.0)
    assert_close(actual, expected)


@mark.parametrize("pad", [my_nn.pad2d])
def test_pad2d_multi_channel(pad) -> None:
    x = torch.arange(4).float().view((1, 2, 2, 1))
    expected = torch.tensor(
        [
            [
                [[-1.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]],
                [[-1.0, 2.0], [-1.0, 3.0], [-1.0, -1.0]],
            ]
        ]
    )
    actual = pad(x, 1, 0, 0, 1, -1.0)
    assert_close(actual, expected)


@mark.parametrize("my_conv", [my_nn.conv2d])
def test_conv2d(my_conv, n_tests=10) -> None:

    for _i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        w = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)

        stride = tuple(np.random.randint(1, 5, size=(2,)))
        padding = tuple(np.random.randint(0, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))

        x = torch.randn((b, ci, h, w))
        weights = torch.randn((co, ci, *kernel_size))

        my_output = my_conv(x, weights, stride=stride, padding=padding)
        torch_output = torch.conv2d(x, weights, stride=stride, padding=padding)
        assert_close(my_output, torch_output, atol=1e-4, rtol=1e-4)


@mark.parametrize("my_maxpool2d", [my_nn.maxpool2d])
def test_maxpool2d(my_maxpool2d, n_tests=20) -> None:

    for _i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 50)
        w = np.random.randint(10, 50)
        ci = np.random.randint(1, 20)

        none_stride = bool(np.random.randint(2))
        stride = None if none_stride else tuple(np.random.randint(1, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        kH, kW = kernel_size
        padding = np.random.randint(0, 1 + kH // 2), np.random.randint(0, 1 + kW // 2)

        x = torch.randn((b, ci, h, w))

        my_output = my_maxpool2d(x, kernel_size, stride=stride, padding=padding)

        torch_output = torch.max_pool2d(
            x,
            kernel_size,
            stride=stride,  # type: ignore (None actually is allowed)
            padding=padding,
        )
        assert_close(my_output, torch_output, atol=1e-4, rtol=1e-4)


@mark.parametrize("MaxPool2d", [my_nn.MaxPool2d])
def test_maxpool2d_module(MaxPool2d) -> None:
    m = MaxPool2d((2, 2), stride=2, padding=0)
    x = torch.arange(16).reshape((1, 1, 4, 4)).float()
    expected = torch.tensor([[5.0, 7.0], [13.0, 15.0]])
    assert (m(x) == expected).all()


@mark.parametrize("Conv2d", [my_nn.Conv2d])
def test_conv2d_module(Conv2d) -> None:
    m = Conv2d(4, 5, (3, 3))
    assert isinstance(
        m.weight, torch.nn.parameter.Parameter
    ), "Weight should be registered a parameter!"
    assert m.weight.nelement() == 4 * 5 * 3 * 3


@mark.parametrize("BatchNorm2d", [my_nn.BatchNorm2d])
def test_batchnorm2d_module(BatchNorm2d) -> None:
    num_features = 2
    bn = BatchNorm2d(num_features)
    assert bn.num_features == num_features
    assert isinstance(bn.weight, torch.nn.parameter.Parameter)
    assert isinstance(bn.bias, torch.nn.parameter.Parameter)
    assert isinstance(bn.running_mean, torch.Tensor)
    assert isinstance(bn.running_var, torch.Tensor)
    assert isinstance(bn.num_batches_tracked, torch.Tensor)


@mark.parametrize("BatchNorm2d", [my_nn.BatchNorm2d])
def test_batchnorm2d_forward(BatchNorm2d) -> None:
    num_features = 2
    bn = BatchNorm2d(num_features)
    assert bn.training
    x = torch.randn((100, num_features, 3, 4))
    out = bn(x)
    assert x.shape == out.shape
    assert_close(
        out.mean(dim=(0, 2, 3)), torch.zeros(num_features), atol=1e-6, rtol=1e-6
    )
    assert_close(out.std(dim=(0, 2, 3)), torch.ones(num_features), rtol=1e-2, atol=1e-2)


@mark.parametrize("BatchNorm2d", [my_nn.BatchNorm2d])
def test_batchnorm2d_running_mean(BatchNorm2d) -> None:
    bn = BatchNorm2d(3, momentum=0.6)
    assert bn.training
    x = torch.arange(12).float().view((2, 3, 2, 1))
    mean = torch.tensor([3.5000, 5.5000, 7.5000])
    num_batches = 20
    for i in range(num_batches):
        bn(x)
        expected_mean = (1 - ((1 - bn.momentum) ** (i + 1))) * mean
        assert_close(bn.running_mean, expected_mean)
    assert bn.num_batches_tracked.item() == num_batches

    bn.eval()
    actual_eval_mean = bn(x).mean((0, 2, 3))
    assert_close(actual_eval_mean, torch.zeros(3))


@mark.parametrize("Flatten", [my_nn.Flatten])
def test_flatten(Flatten) -> None:
    x = torch.arange(24).reshape((2, 3, 4))
    assert Flatten(start_dim=0)(x).shape == (24,)
    assert Flatten(start_dim=1)(x).shape == (2, 12)
    assert Flatten(start_dim=0, end_dim=1)(x).shape == (6, 4)
    assert Flatten(start_dim=0, end_dim=-2)(x).shape == (6, 4)


@mark.parametrize("Flatten", [my_nn.Flatten])
def test_flatten_is_view(Flatten) -> None:
    x = torch.arange(24).reshape((2, 3, 4))
    view = Flatten()(x)
    view[0][0] = 99
    assert x[0, 0, 0] == 99


@mark.parametrize("Linear", [my_nn.Linear])
def test_linear_forward(Linear) -> None:
    x = torch.rand((10, 512))
    yours = Linear(512, 64)

    assert yours.weight.shape == (64, 512)
    assert yours.bias.shape == (64,)

    official = torch.nn.Linear(512, 64)
    yours.weight = official.weight
    yours.bias = official.bias
    actual = yours(x)
    expected = official(x)
    assert_close(actual, expected)


@mark.parametrize("Linear", [my_nn.Linear])
def test_linear_parameters(Linear) -> None:
    m = Linear(2, 3)
    params = dict(m.named_parameters())
    assert len(params) == 2, f"Your model has {len(params)} recognized Parameters"
    assert list(params.keys()) == [
        "weight",
        "bias",
    ], "For compatibility with PyTorch, your fields should be named weight and bias."


@mark.parametrize("Linear", [my_nn.Linear])
def test_linear_no_bias(Linear) -> None:
    m = Linear(3, 4, bias=False)
    assert m.bias is None, "Bias should be None when not enabled."
    assert len(list(m.parameters())) == 1


@mark.parametrize("Sequential", [my_nn.Sequential])
def test_sequential(Sequential) -> None:

    modules = [torch.nn.Linear(1, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1)]
    s = Sequential(*modules)

    assert list(s.modules()) == [
        s,
        *modules,
    ], "The sequential and its submodules should be registered Modules."
    assert (
        len(list(s.parameters())) == 4
    ), "Submodules's parameters should be registered."


@mark.parametrize("Sequential", [my_nn.Sequential])
def test_sequential_forward(Sequential) -> None:

    modules = [torch.nn.Linear(1, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1)]
    x = torch.tensor([5.0])
    s = Sequential(*modules)
    actual_out = s(x)
    expected_out = modules[-1](modules[-2](modules[-3](x)))
    assert_close(actual_out, expected_out)


@mark.parametrize("ReLU", [my_nn.ReLU])
def test_relu(ReLU) -> None:
    input = torch.rand((10, 50, 20, 30))
    assert torch.equal(ReLU()(input), torch.nn.ReLU()(input))


@mark.parametrize("AveragePool", [my_nn.AveragePool])
def test_avg_pool(AveragePool) -> None:
    input = torch.rand((10, 50, 20, 30))
    assert torch.equal(
        AveragePool()(input), torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))(input)
    )
