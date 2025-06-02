import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | tuple | torch.Size,
        eps=1e-5,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.normalize_dims = tuple(range(-1, -1 - len(self.normalized_shape), -1))
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(normalized_shape, device=device, dtype=dtype)
            )
            self.bias = nn.Parameter(
                torch.empty(normalized_shape, device=device, dtype=dtype)
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        assert x.shape[-len(self.normalized_shape) :] == self.normalized_shape
        mean = x.mean(dim=self.normalize_dims, keepdim=True)
        var = x.var(dim=self.normalize_dims, unbiased=False, keepdim=True)
        out = (x - mean) / (var + self.eps) ** 0.5
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.weight[x]
