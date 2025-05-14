from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Concatenate, ParamSpec, Self

import numpy as np

Array = np.ndarray

grad_tracking_enabled = True


@dataclass
class Recipe:
    """Extra information necessary to run backpropagation.

    If recipe is not None, this tensor was created via
    recipe.func(*recipe.args, **recipe.kwargs)
    """

    func: Callable
    "The 'inner' NumPy function that does the actual forward computation"
    args: tuple[Any, ...]
    "The unwrapped input arguments, raw NumPy arrays"
    kwargs: dict[str, Any]
    "Keyword arguments."
    parents: dict[int, "Tensor"]
    "Map from positional argument index to the Tensor at that position."


class Tensor:
    """A drop-in replacement for torch.Tensor supporting a subset of features."""

    array: Array
    "The underlying array. Can be shared between multiple Tensors."
    requires_grad: bool
    "Should track relevant data for backprop."
    grad: Self | None
    "Backpropagation will accumulate gradients into this field."
    recipe: Recipe | None
    "Extra information necessary to run backpropagation."

    def __init__(self, array: Array | list, requires_grad=False) -> None:
        self.array = array if isinstance(array, Array) else np.array(array)
        self.requires_grad = requires_grad
        self.grad = None
        self.recipe = None

    def __neg__(self) -> "Tensor":
        return negative(self)

    def __add__(self, other) -> "Tensor":
        return add(self, other)

    def __radd__(self, other) -> "Tensor":
        return add(other, self)

    def __sub__(self, other) -> "Tensor":
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other) -> "Tensor":
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __truediv__(self, other):
        return true_divide(self, other)

    def __rtruediv__(self, other):
        return true_divide(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __eq__(self, other):
        return eq(self, other)

    def __repr__(self) -> str:
        return f"Tensor({self.array!r}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        if self.array.ndim == 0:
            raise TypeError
        return self.array.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def __getitem__(self, index: "Index") -> "Tensor":
        return getitem(self, index)

    def add_(self, other: "Tensor", alpha: float = 1.0) -> "Tensor":
        add_(self, other, alpha=alpha)
        return self

    def T(self) -> "Tensor":
        return permute(self)

    def item(self):
        return self.array.item()

    def sum(self, dim: None | int | Iterable[int] = None, keepdim=False):
        return sum(self, dim=dim, keepdim=keepdim)

    def log(self):
        return log(self)

    def exp(self):
        return exp(self)

    def reshape(self, new_shape: tuple[int]):
        return reshape(self, new_shape)

    def expand(self, new_shape: tuple[int]):
        return expand(self, new_shape)

    def permute(self, dims: Iterable[int]):
        return permute(self, dims)

    def maximum(self, other):
        return maximum(self, other)

    def relu(self):
        return relu(self)

    def argmax(self, dim: None | int = None, keepdim=False):
        return argmax(self, dim=dim, keepdim=keepdim)

    def uniform_(self, low: float, high: float) -> "Tensor":
        self.array[:] = np.random.uniform(low, high, self.array.shape)
        return self

    def backward(self, end_grad: Array | Self | None = None) -> None:
        if isinstance(end_grad, Array):
            end_grad = Tensor(end_grad)
        return backprop(self, end_grad)

    def size(self, dim: int | None = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def is_leaf(self):
        """See https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html."""
        return not self.requires_grad or self.recipe is None or not self.recipe.parents


def get_recipe(tensor: Tensor, name="target", skip_trivial=False) -> dict[str, Tensor]:
    recipe = {}
    if (
        skip_trivial
        and hasattr(tensor, "recipe")
        and tensor.recipe
        and len(tensor.recipe.parents) == 1
    ):
        pass
    else:
        recipe[name] = tensor
    if hasattr(tensor, "recipe") and tensor.recipe:
        for k, parent in tensor.recipe.parents.items():
            func_name = tensor.recipe.func.__name__
            if len(tensor.recipe.parents) > 1:
                func_name += f"[{k}]"
            recipe.update(
                get_recipe(parent, f"{name}.{func_name}", skip_trivial=skip_trivial)
            )
    return recipe


def empty(*shape: int) -> Tensor:
    """Like torch.empty."""
    return Tensor(np.empty(shape))


def zeros(*shape: int) -> Tensor:
    """Like torch.zeros."""
    return Tensor(np.zeros(shape))


def arange(start: int, end: int, step=1) -> Tensor:
    """Like torch.arange(start, end)."""
    return Tensor(np.arange(start, end, step=step))


def tensor(array: Array, requires_grad=False) -> Tensor:
    """Like torch.tensor."""
    return Tensor(array, requires_grad=requires_grad)


def topological_sort(node: Tensor) -> list[Tensor]:
    """Return a list of node's descendants in reverse topological order (future to past).

    Use the depth-first search
    """
    perm: set[Tensor] = set()
    temp: set[Tensor] = set()
    result: list[Tensor] = []

    def visit(cur: Tensor) -> None:
        if cur in perm:
            return
        if cur in temp:
            raise ValueError("Not a DAG!")
        temp.add(cur)

        if cur.recipe is not None:
            for prev in cur.recipe.parents.values():
                visit(prev)

        temp.remove(cur)
        perm.add(cur)
        result.append(cur)

    visit(node)
    return list(reversed(result))


def backprop(end_node: Tensor, end_grad: Tensor | None = None) -> None:
    """Accumulate gradients in the grad field of each leaf node.

    tensor.backward() is equivalent to backprop(tensor).

    end_node: the rightmost node in the computation graph
    end_grad: the grad of the loss wrt end_node: all 1s if not specified.
    """
    grads = {}
    grads[end_node] = (
        np.ones_like(end_node.array) if end_grad is None else end_grad.array
    )
    assert isinstance(grads[end_node], Array)
    for node in topological_sort(end_node):
        assert node in grads
        grad_out = grads.pop(node)
        if node.is_leaf:
            if node.grad is None:
                node.grad = Tensor(grad_out)
            else:
                node.grad.array += grad_out
            assert isinstance(node.grad, Tensor)
            assert isinstance(node.grad.array, Array)
        else:
            assert node.recipe
            back_funcs = BACK_FUNCS[node.recipe.func]
            for arg, back_func in back_funcs.items():
                if arg not in node.recipe.parents:
                    # the back_func defines it but the recipe don't have it
                    # we assume it's a non-tensor, i.e in the forward func `x * 2`
                    continue
                assert isinstance(grad_out, Array)
                grad_in = back_func(
                    grad_out, node.array, *node.recipe.args, **node.recipe.kwargs
                )
                assert isinstance(grad_in, Array)
                if node.recipe.parents[arg] in grads:
                    grads[node.recipe.parents[arg]] += grad_in
                else:
                    grads[node.recipe.parents[arg]] = grad_in


def wrap(
    numpy_func: Callable[..., Array], is_differentiable=True
) -> Callable[..., Tensor]:
    """
    numpy_func: function taking some number of NumPy arrays as positional arguments
     and returning NumPy array.
    is_differentiable: if True, numpy_func is differentiable with respect to some input
     argument, so we may need to track information. If False, we definitely don't need
     to track information.

    Return: function taking some number of Tensor and returning Tensor.
    """

    def tensor_func(*args: Any, **kwargs: Any) -> Tensor:
        numpy_args = tuple(x.array if isinstance(x, Tensor) else x for x in args)
        if not numpy_args:
            raise ValueError("no args. did you use kwargs by mistake?")
        output_array = numpy_func(*numpy_args, **kwargs)
        requires_grad = (
            grad_tracking_enabled
            and is_differentiable
            and any(
                x.requires_grad or (x.recipe is not None)
                for x in args
                if isinstance(x, Tensor)
            )
        )
        output = Tensor(output_array, requires_grad=requires_grad)
        if requires_grad:
            output.recipe = Recipe(
                func=numpy_func,
                args=numpy_args,
                kwargs=kwargs,
                parents={i: x for i, x in enumerate(args) if isinstance(x, Tensor)},
            )
        return output

    return tensor_func


P = ParamSpec("P")
BACK_FUNCS: defaultdict[
    Callable[P, Array], dict[int, Callable[Concatenate[Array, Array, P], Array]]
] = defaultdict(dict)


## Non-Differentiable Functions
def _argmax(x: Array, dim: None | int = None, keepdim=False):
    """Like torch.argmax."""
    return np.argmax(x, axis=dim, keepdims=keepdim)


argmax = wrap(_argmax, is_differentiable=False)
eq = wrap(np.equal, is_differentiable=False)


## Differentiable Functions


def log_back(grad_out: Array, out: Array, x: Array) -> Array:
    # out = log(x)
    # dl/dx = dl/do do/dx = grad_out * 1/x
    return grad_out / x


log = wrap(np.log)

BACK_FUNCS[np.log][0] = log_back


def negative_back(grad_out: Array, out: Array, x: Array) -> Array:
    return -grad_out


negative = wrap(np.negative)
BACK_FUNCS[np.negative][0] = negative_back


def exp_back(grad_out: Array, out: Array, x: Array) -> Array:
    # out = exp(x)
    # dl/dx = dl/do do/dx = grad_out * exp(x)
    return grad_out * out


exp = wrap(np.exp)
BACK_FUNCS[np.exp][0] = exp_back


## Reshape / Permute


def reshape_back(grad_out: Array, out: Array, x: Array, new_shape: tuple[int]) -> Array:
    return grad_out.reshape(x.shape)


reshape = wrap(np.reshape)
BACK_FUNCS[np.reshape][0] = reshape_back


def permute_back(grad_out: Array, out: Array, x: Array, axes: Iterable[int]) -> Array:
    bwd_transpose = {j: i for i, j in enumerate(axes)}
    bwd_axes = [bwd_transpose[k] for k in range(len(bwd_transpose))]
    return grad_out.transpose(bwd_axes)


BACK_FUNCS[np.transpose][0] = permute_back
permute = wrap(np.transpose)


## Broadcasting
def unbroadcast(broadcasted: Array, original: Array) -> Array:
    """Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original'.
    """
    if original.ndim == 0:
        # regular code will not return Arr in this case
        return np.array(broadcasted.sum())
    while broadcasted.ndim > original.ndim:
        broadcasted = broadcasted.sum(0)
    assert broadcasted.ndim == original.ndim
    for dim, size in enumerate(original.shape):
        if size == 1:
            broadcasted = broadcasted.sum(dim, keepdims=True)
    return broadcasted


def expand_back(grad_out: Array, out: Array, x: Array, new_shape: tuple[int]) -> Array:
    return unbroadcast(grad_out, x)


def _expand(x: Array, new_shape: tuple[int]) -> Array:
    """Like torch.expand, calling np.broadcast_to internally.

    Note torch.expand supports -1 for a dimension size meaning "don't change the size".
    np.broadcast_to does not natively support this.
    """
    shape = x.shape
    new_shape_left, new_shape_right = new_shape[: -len(shape)], new_shape[-len(shape) :]
    new_shape_right = tuple(
        new if new >= 0 else old
        for new, old in zip(new_shape_right, shape, strict=True)
    )
    return np.broadcast_to(x, new_shape_left + new_shape_right)


expand = wrap(_expand)
BACK_FUNCS[_expand][0] = expand_back


def sum_back(
    grad_out: Array,
    out: Array,
    x: Array,
    dim: None | int | Iterable[int] = None,
    keepdim=False,
):
    assert isinstance(grad_out, Array)
    if dim is None:
        dim = list(range(x.ndim))
    elif isinstance(dim, int):
        dim = [dim]
    dim = [d % x.ndim for d in dim]
    bc_shape = (
        tuple(1 if d in dim else s for d, s in enumerate(x.shape))
        if not keepdim
        else out.shape
    )
    assert out.shape == grad_out.shape
    return_dims = grad_out.reshape(bc_shape)
    return np.broadcast_to(return_dims, x.shape)


def _sum(x: Array, dim: None | int | Iterable[int] = None, keepdim=False) -> Array:
    """Like torch.sum, calling np.sum internally."""
    return np.sum(x, axis=dim, keepdims=keepdim)


sum = wrap(_sum)
BACK_FUNCS[_sum][0] = sum_back

## Indexing

Index = int | tuple[int | Array | Tensor, ...]


def index_tensor_to_np(index: Index) -> int | tuple[int | Array, ...]:
    return (
        tuple(t.array if isinstance(t, Tensor) else t for t in index)
        if isinstance(index, tuple)
        else index
    )


def _getitem(x: Array, index: Index) -> Array:
    """Like x[index] when x is a torch.Tensor."""
    return x[index_tensor_to_np(index)]


def getitem_back(grad_out: Array, out: Array, x: Array, index: Index):
    """Backwards function for _getitem.

    Hint: use np.add.at(a, indices, b)
    """
    index = index_tensor_to_np(index)
    grad_in = np.zeros_like(x)
    np.add.at(grad_in, index, grad_out)
    return grad_in


getitem = wrap(_getitem)
BACK_FUNCS[_getitem][0] = getitem_back


## Functions of Two Tensors


def multiply_back0(grad_out: Array, out: Array, x: Array, y: Any) -> Array:
    """Backwards function for x * y wrt argument 0 aka x."""
    return unbroadcast(grad_out * y, x)


def multiply_back1(grad_out: Array, out: Array, x: Any, y: Array) -> Array:
    """Backwards function for x * y wrt argument 1 aka y."""
    return unbroadcast(grad_out * x, y)


multiply = wrap(np.multiply)
BACK_FUNCS[np.multiply][0] = multiply_back0
BACK_FUNCS[np.multiply][1] = multiply_back1

add = wrap(np.add)
subtract = wrap(np.subtract)
true_divide = wrap(np.true_divide)
BACK_FUNCS[np.add][0] = lambda grad_out, out, *args: unbroadcast(
    grad_out, np.array(args[0])
)
BACK_FUNCS[np.add][1] = lambda grad_out, out, *args: unbroadcast(
    grad_out, np.array(args[1])
)
BACK_FUNCS[np.subtract][0] = lambda grad_out, out, *args: unbroadcast(
    grad_out, np.array(args[0])
)
BACK_FUNCS[np.subtract][1] = lambda grad_out, out, *args: unbroadcast(
    -grad_out, np.array(args[1])
)
BACK_FUNCS[np.true_divide][0] = lambda grad_out, out, *args: unbroadcast(
    grad_out / np.array(args[1]), np.array(args[0])
)
BACK_FUNCS[np.true_divide][1] = lambda grad_out, out, *args: unbroadcast(
    -grad_out * out / np.array(args[1]), np.array(args[1])
)


## In-Place Operations [beware! silently compute the wrong gradients]


def add_(x: Tensor, other: Tensor, alpha: float = 1.0) -> Tensor:
    """Like torch.add_. Compute x += other * alpha in-place and return tensor."""
    np.add(x.array, other.array * alpha, out=x.array)
    return x


## Splitting Gradients - elementwise maximum
def maximum_back0(grad_out: Array, out: Array, x: Array, y: Array) -> Array:
    """Backwards function for max(x, y) wrt x."""
    return unbroadcast(
        np.where(x > y, grad_out, np.where(x == y, 0.5 * grad_out, 0)), x
    )


def maximum_back1(grad_out: Array, out: Array, x: Array, y: Array) -> Array:
    """Backwards function for max(x, y) wrt y."""
    return unbroadcast(
        np.where(x < y, grad_out, np.where(x == y, 0.5 * grad_out, 0)), y
    )


maximum = wrap(np.maximum)
BACK_FUNCS[np.maximum][0] = maximum_back0
BACK_FUNCS[np.maximum][1] = maximum_back1


## Functional ReLU
def relu(x: Tensor) -> Tensor:
    """Like torch.nn.function.relu(x, inplace=False)."""
    return x.maximum(0)


## 2D Matrix Multiply
def _matmul2d(x: Array, y: Array) -> Array:
    """Matrix multiply restricted to the case where both inputs are exactly 2D."""
    return x @ y


def matmul2d_back0(grad_out: Array, out: Array, x: Array, y: Array) -> Array:
    # x  @  y -> out
    # ij , jk -> ik
    # out_ik = sum{j} x_ij * y_jk
    # grad_x_ij = grad{ik} * y_jk // sum k
    return grad_out @ y.T


def matmul2d_back1(grad_out: Array, out: Array, x: Array, y: Array) -> Array:
    # grad_y_jk = x_ij * grad{ik} // sum i
    return x.T @ grad_out


matmul = wrap(_matmul2d)
BACK_FUNCS[_matmul2d][0] = matmul2d_back0
BACK_FUNCS[_matmul2d][1] = matmul2d_back1


## `nn.Parameter`
class Parameter(Tensor):
    def __init__(self, tensor: Tensor, requires_grad=True) -> None:
        """Share the array with the provided tensor."""
        super().__init__(tensor.array, requires_grad=requires_grad)

    def __repr__(self) -> str:
        return f"Parameter containing:\n{super().__repr__()}"


## Build Your Own `no_grad`
class NoGrad:
    """Context manager that disables grad inside the block. Like torch.no_grad."""

    was_enabled: bool

    def __enter__(self):
        global grad_tracking_enabled
        self.was_enabled = grad_tracking_enabled
        grad_tracking_enabled = False

    def __exit__(self, type, value, traceback):
        global grad_tracking_enabled
        grad_tracking_enabled = self.was_enabled
