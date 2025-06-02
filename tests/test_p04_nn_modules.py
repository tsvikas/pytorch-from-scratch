import torch
import torch.nn as nn
from torch.testing import assert_close

from pytorch_from_scratch.p04_BERT.my_nn_modules import Embedding, LayerNorm


def test_layernorm_mean_1d():
    """If an integer is passed, this means normalize over the last dimension which should have that size."""
    x = torch.randn(20, 10)
    ln1 = LayerNorm(10)
    out = ln1(x)
    max_mean = out.mean(-1).abs().max().item()
    assert max_mean < 1e-5, f"Normalized mean should be about 0, got {max_mean}"


def test_layernorm_mean_2d():
    """If normalized_shape is 2D, should normalize over both the last two dimensions."""
    x = torch.randn(20, 10)
    ln1 = LayerNorm((20, 10))
    out = ln1(x)
    max_mean = out.mean((-1, -2)).abs().max().item()
    assert max_mean < 1e-5, f"Normalized mean should be about 0, got {max_mean}"


def test_layernorm_std():
    """If epsilon is small enough and no elementwise_affine, the output variance should be very close to 1."""
    x = torch.randn(20, 10)
    ln1 = LayerNorm(10, eps=1e-11, elementwise_affine=False)
    out = ln1(x)
    var_diff = (1 - out.var(-1, unbiased=False)).abs().max().item()
    assert var_diff < 1e-6, f"Var should be about 1, off by {var_diff}"


def test_layernorm_exact():
    """Your LayerNorm's output should exactly match PyTorch for equal epsilon."""
    x = torch.randn(2, 3, 4, 5)
    # Use large epsilon to make sure it fails if they forget it
    ln1 = LayerNorm((5,), eps=1e-2)
    ln2 = torch.nn.LayerNorm((5,), eps=1e-2)  # type: ignore
    actual = ln1(x)
    expected = ln2(x)
    assert_close(actual, expected)


def test_layernorm_backward():
    """The backwards pass should also match PyTorch exactly."""
    x = torch.randn(10, 3)
    x2 = x.clone()
    x.requires_grad_(True)
    x2.requires_grad_(True)

    # Without parameters, should be deterministic
    ref = nn.LayerNorm(3, elementwise_affine=False)
    ref.requires_grad_(True)
    ref(x).sum().backward()

    ln = LayerNorm(3, elementwise_affine=False)
    ln.requires_grad_(True)
    ln(x2).sum().backward()
    # Use atol since grad entries are supposed to be zero here
    assert isinstance(x.grad, torch.Tensor)
    assert isinstance(x2.grad, torch.Tensor)
    assert_close(x.grad, x2.grad, atol=1e-5, rtol=1e-5)


def test_embedding():
    """Indexing into the embedding should fetch the corresponding rows of the embedding."""
    emb = Embedding(6, 10)
    out = emb(torch.LongTensor([1, 3, 5]))
    assert_close(out[0], emb.weight[1])
    assert_close(out[1], emb.weight[3])
    assert_close(out[2], emb.weight[5])
