import numpy as np
import pytest

from my_tensor import (
    NoGrad,
    Parameter,
    Tensor,
    backprop,
    log,
    log_back,
    topological_sort,
    unbroadcast,
)


def test_tensor_by_keyword() -> None:
    with pytest.raises(ValueError):
        log(x=Tensor([100]))


def test_requires_grad_true() -> None:
    a = Tensor([1.0, 0.0, 3.0, 4.0], requires_grad=True)
    b = a.argmax()
    assert not b.requires_grad
    assert b.recipe is None
    assert b.item() == 3


def test_safe_example() -> None:
    """This example should work properly."""
    a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
    a.add_(b)
    c = a * b
    c.sum().backward()
    assert a.grad is not None
    assert np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0])
    assert b.grad is not None
    assert np.allclose(b.grad.array, [2.0, 4.0, 6.0, 8.0])


def test_unsafe_example() -> None:
    """This example is expected to compute the wrong gradients."""
    a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
    c = a * b
    a.add_(b)
    c.sum().backward()
    assert np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0])
    assert not np.allclose(b.grad.array, [0.0, 1.0, 2.0, 3.0])


def test_grad() -> None:
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    backprop(a * 2)
    b = Tensor([0, 1, 2, 3], requires_grad=True)
    backprop(2 * b)
    assert a.grad is not None
    assert b.grad is not None
    assert np.allclose(a.grad.array, b.grad.array)


def test_parameter() -> None:
    x = Tensor([1.0, 2.0, 3.0])
    p = Parameter(x)
    assert p.requires_grad
    assert p.array is x.array
    assert (
        repr(p)
        == "Parameter containing:\nTensor(array([1., 2., 3.]), requires_grad=True)"
    )
    x.add_(Tensor(np.array(2.0)))
    assert np.allclose(p.array, np.array([3.0, 4.0, 5.0]))


def test_log() -> None:
    a = Tensor([np.e, np.e**np.e], requires_grad=True)
    b = log(a)
    assert np.allclose(b.array, [1, np.e])
    assert b.requires_grad is True, "Should require grad because input required grad."
    assert b.is_leaf is False
    assert b.recipe is not None
    assert len(b.recipe.parents) == 1
    assert b.recipe.parents[0] is a
    assert len(b.recipe.args) == 1
    assert b.recipe.args[0] is a.array
    assert b.recipe.kwargs == {}
    assert b.recipe.func is np.log
    c = log(b)
    assert np.allclose(c.array, [0, 1])

    d = Tensor([1, np.e])
    e = log(d)
    assert e.requires_grad is False, "Should not require grad because input did not."
    assert e.recipe is None
    assert np.allclose(e.array, [0, 1])


def test_topological_sort() -> None:
    a = Tensor([np.e, np.e**np.e], requires_grad=True)
    b = a.log()
    c = b.log()
    assert all(
        np.allclose(i1.array, i2.array)
        for i1, i2 in zip(topological_sort(c), [c, b, a], strict=True)
    )


def test_log_back() -> None:
    a = Tensor([1, np.e, np.e**np.e], requires_grad=True)
    b = a.log()
    grad_out = np.array([2.0, 2.0, 2.0])
    actual = log_back(grad_out, b.array, a.array)
    expected = [2.0, 2.0 / np.e, 2.0 / (np.e**np.e)]
    assert np.allclose(actual, expected)


def test_backprop() -> None:
    a = Tensor([np.e, np.e**np.e], requires_grad=True)
    b = a.log()
    c = b.log()
    c.backward()
    assert c.grad is None
    assert b.grad is None
    assert a.grad is not None
    assert np.allclose(a.grad.array, 1 / b.array / a.array)


def test_negative_back() -> None:
    a = Tensor([-1, 0, 1], requires_grad=True)
    b = -a
    c = -b
    c.backward()
    assert a.grad is not None
    assert np.allclose(a.grad.array, [1, 1, 1])


def test_exp_back() -> None:
    a = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    b = a.exp()
    b.backward()
    assert a.grad is not None
    assert np.allclose(a.grad.array, 1 / np.e, 0, np.e)

    a = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    b = a.exp()
    c = b.exp()
    c.backward()

    def d(x):
        return (np.e**x) * (np.e ** (np.e**x))

    assert a.grad is not None
    assert np.allclose(a.grad.array, *[d(x) for x in a.array])


def test_reshape_back() -> None:
    a = Tensor([1, 2, 3, 4, 5, 6], requires_grad=True)
    b = a.reshape((3, 2))
    b.backward()
    assert a.grad is not None
    assert np.allclose(a.grad.array, np.ones(6))


def test_permute_back() -> None:
    a = Tensor(np.arange(24).reshape((2, 3, 4)), requires_grad=True)
    out = a.permute((2, 0, 1))
    out.backward(np.arange(24).reshape((4, 2, 3)))

    assert a.grad is not None
    assert np.allclose(
        a.grad.array,
        np.array(
            [
                [
                    [0.0, 6.0, 12.0, 18.0],
                    [1.0, 7.0, 13.0, 19.0],
                    [2.0, 8.0, 14.0, 20.0],
                ],
                [
                    [3.0, 9.0, 15.0, 21.0],
                    [4.0, 10.0, 16.0, 22.0],
                    [5.0, 11.0, 17.0, 23.0],
                ],
            ]
        ),
    )


def test_unbroadcast() -> None:
    small = np.ones((2, 1, 3))
    large = np.broadcast_to(small, (5, 1, 2, 4, 3))
    out = unbroadcast(large, small)
    assert out.shape == small.shape
    assert (
        out == 20.0
    ).all(), "Each element in the small array appeared 20 times in the large array."

    small = np.ones((2, 1, 3))
    large = np.broadcast_to(small, (5, 1, 2, 1, 3))
    out = unbroadcast(large, small)
    assert out.shape == small.shape
    assert (
        out == 5.0
    ).all(), "Each element in the small array appeared 5 times in the large array."

    small = np.ones((2, 1, 3))
    large = np.broadcast_to(small, (2, 4, 3))
    out = unbroadcast(large, small)
    assert out.shape == small.shape
    assert (
        out == 4.0
    ).all(), "Each element in the small array appeared 4 times in the large array."


def test_expand() -> None:
    a = Tensor(np.ones((2, 1, 3)), requires_grad=True)
    b = a.expand((5, 1, 2, 4, 3))
    b.backward(np.full_like(b.array, 10.0))
    assert a.grad is not None
    assert a.grad.shape == a.array.shape
    assert (a.grad.array == 20 * 10.0).all()


def test_expand_negative_length() -> None:
    a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    b = a.expand((3, 2, -1))
    assert b.shape == (3, 2, 5)
    b.backward()
    assert a.grad is not None
    assert a.grad.shape == a.array.shape
    assert (a.grad.array == 6).all()


def test_sum_keepdim_false() -> None:
    a = Tensor(
        np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]),
        requires_grad=True,
    )
    b = a.sum(0)
    c = b.sum(0)
    c.backward(np.array(2))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 2).all()


def test_sum_keepdim_true() -> None:
    a = Tensor(
        np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]),
        requires_grad=True,
    )
    c = a.sum(0, keepdim=True)
    assert np.allclose(c.array, np.array([[5.0, 7.0, 9.0, 11.0, 13.0]]))
    c.backward()
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 1).all()


def test_sum_dim_none() -> None:
    a = Tensor(
        np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]),
        requires_grad=True,
    )
    b = a.sum()
    b.backward(np.array(4))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 4).all()


def test_getitem_int() -> None:
    a = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
    b = a[1]
    c = b.sum(0)
    c.backward(np.array(10.0))
    assert a.grad is not None
    assert np.allclose(a.grad.array, np.array([[0, 0, 0], [10, 10, 10]]))


def test_getitem_tuple() -> None:
    a = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
    b = a[(1, 2)]
    b.backward(np.array(10.0))
    assert a.grad is not None
    assert np.allclose(a.grad.array, np.array([[0, 0, 0], [0, 0, 10]]))


def test_getitem_integer_array() -> None:
    a = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
    index = np.array([0, 1, 0, 1, 0]), np.array([0, 0, 1, 2, 0])
    out = a[index]
    out.sum().backward(np.array(10.0))
    assert a.grad is not None
    assert np.allclose(a.grad.array, np.array([[20, 10, 0], [10, 0, 10]]))


def test_getitem_integer_tensor() -> None:
    a = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
    index = Tensor(np.array([0, 1, 0, 1, 0])), Tensor(np.array([0, 0, 1, 2, 0]))
    out = a[index]
    out.sum().backward(np.array(10.0))
    assert a.grad is not None
    assert np.allclose(a.grad.array, np.array([[20, 10, 0], [10, 0, 10]]))


def test_multiply_broadcasted() -> None:
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = Tensor([[0], [1], [10]], requires_grad=True)
    c = a * b
    c.backward()
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 11).all()
    assert b.grad is not None
    assert b.grad.shape == b.shape
    assert (b.grad.array == 6).all()


def test_add_broadcasted() -> None:
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = Tensor([[0], [1], [10]], requires_grad=True)
    c = a + b
    c.backward()
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 3).all()
    assert b.grad is not None
    assert b.grad.shape == b.shape
    assert (b.grad.array == 4).all()


def test_subtract_broadcasted() -> None:
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = Tensor([[0], [1], [10]], requires_grad=True)
    c = a - b
    c.backward()
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 3).all()
    assert b.grad is not None
    assert b.grad.shape == b.shape
    assert (b.grad.array == -4).all()


def test_truedivide_broadcasted() -> None:
    a = Tensor([0, 6, 12, 18], requires_grad=True)
    b = Tensor([[1], [2], [3]], requires_grad=True)
    c = a / b
    c.backward()
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == (1 + 1 / 2 + 1 / 3)).all()
    assert b.grad is not None
    assert b.grad.shape == b.shape
    assert np.equal(b.grad.array, np.array([[-36.0], [-9.0], [-4.0]])).all()


def test_maximum() -> None:
    a = Tensor([0, 1, 2], requires_grad=True)
    b = Tensor([-1, 1, 3], requires_grad=True)
    out = a.maximum(b)
    assert np.allclose(out.array, [0, 1, 3])
    out.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert np.allclose(a.grad.array, [1, 0.5, 0])
    assert np.allclose(b.grad.array, [0, 0.5, 1])


def test_maximum_broadcasted() -> None:
    a = Tensor([0, 1, 2], requires_grad=True)
    b = Tensor([[-1], [1], [3]], requires_grad=True)
    out = a.maximum(b)
    assert np.allclose(out.array, np.array([[0, 1, 2], [1, 1, 2], [3, 3, 3]]))
    out.backward()
    assert a.grad is not None
    assert np.allclose(a.grad.array, np.array([1.0, 1.5, 2.0]))
    assert b.grad is not None
    assert np.allclose(b.grad.array, np.array([[0.0], [1.5], [3.0]]))


def test_relu() -> None:
    a = Tensor([-1, 0, 1], requires_grad=True)
    out = a.relu()
    out.backward()
    assert a.grad is not None
    assert np.allclose(a.grad.array, np.array([0, 0.5, 1.0]))


def test_matmul2d() -> None:
    a = Tensor(np.arange(-3, 3).reshape((2, 3)), requires_grad=True)
    b = Tensor(np.arange(-4, 5).reshape((3, 3)), requires_grad=True)
    out = a @ b
    out.backward()
    assert a.grad is not None
    assert b.grad is not None
    assert np.allclose(a.grad.array, np.array([[-9, 0, 9], [-9, 0, 9]]))
    assert np.allclose(b.grad.array, np.array([[-3, -3, -3], [-1, -1, -1], [1, 1, 1]]))


def test_no_grad() -> None:
    a = Tensor([1], requires_grad=True)
    with NoGrad():
        b = a + a
    c = a + a
    assert b.requires_grad is False
    assert b.recipe is None
    assert c.requires_grad
    assert c.recipe is not None


def test_no_grad_nested() -> None:
    a = Tensor([1], requires_grad=True)
    with NoGrad(), NoGrad(), NoGrad():
        b = a + a
    assert b.requires_grad is False
    assert b.recipe is None
