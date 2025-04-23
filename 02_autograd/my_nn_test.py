import warnings

import numpy as np

from my_nn import Module, cross_entropy
from my_tensor import Parameter, Tensor


def test_cross_entropy() -> None:
    logits = Tensor(
        [
            [float("-inf"), float("-inf"), 0],
            [1 / 3, 1 / 3, 1 / 3],
            [float("-inf"), 0, 0],
        ]
    )
    true_labels = Tensor([2, 0, 0])
    expected = Tensor([0.0, np.log(3), float("inf")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = cross_entropy(logits, true_labels)
    assert np.allclose(actual.array, expected.array)


def test_module() -> None:
    class TestInnerModule(Module):
        def __init__(self) -> None:
            super().__init__()
            self.param1 = Parameter(Tensor([1.0]))
            self.param2 = Parameter(Tensor([2.0]))

    class TestModule(Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = TestInnerModule()
            self.param3 = Parameter(Tensor([3.0]))

    mod = TestModule()
    assert list(mod.modules()) == [mod.inner]
    assert list(mod.parameters()) == [mod.param3, mod.inner.param1, mod.inner.param2]
