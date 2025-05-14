from torch import Tensor
from torch.nn import Module

from .my_nn import (
    AveragePool,
    BatchNorm2d,
    Conv2d,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)


class ResidualBlock(Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1) -> None:
        """A single residual block with optional downsampling.

        If first_stride is > 1, this means the optional (conv + bn)
        should be present on the right branch
        """
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = first_stride
        self.left_branch = Sequential(
            Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, kernel_size=3, padding=1),
            BatchNorm2d(out_feats),
        )
        if first_stride > 1:
            self.right_branch = Sequential(
                Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride),
                BatchNorm2d(out_feats),
            )
        else:
            self.right_branch = Sequential()
        self.combine = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)
        """
        left = self.left_branch(x)
        right = self.right_branch(x)
        return self.combine(left + right)


class BlockGroup(Module):
    def __init__(
        self, n_blocks: int, in_feats: int, out_feats: int, first_stride: int = 1
    ) -> None:
        """An n_blocks sequence of ResidualBlock.

        Only the first block uses the provided stride.
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = first_stride

        self.blocks = Sequential(
            *[
                ResidualBlock(
                    in_feats if i == 0 else out_feats,
                    out_feats,
                    first_stride if i == 0 else 1,
                )
                for i in range(n_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute the forward pass.
        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        """
        return self.blocks(x)


class ResNet34(Module):
    def __init__(
        self,
        n_blocks_per_group=(3, 4, 6, 3),
        out_features_per_group=(64, 128, 256, 512),
        strides_per_group=(1, 2, 2, 2),
        n_classes=1000,
    ) -> None:
        super().__init__()
        first_in_feats = 3
        first_out_feats = 64
        self.n_blocks_per_group = list(n_blocks_per_group)
        self.out_features_per_group = list(out_features_per_group)
        self.strides_per_group = list(strides_per_group)
        self.n_classes = n_classes

        block_groups = []
        in_feats = first_out_feats
        for n_blocks, out_features, strides in zip(
            n_blocks_per_group, out_features_per_group, strides_per_group, strict=False
        ):
            block_groups.append(BlockGroup(n_blocks, in_feats, out_features, strides))
            in_feats = out_features

        self.blocks = Sequential(
            # N C H W
            Conv2d(first_in_feats, first_out_feats, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(first_out_feats),
            ReLU(),
            MaxPool2d(3, padding=1, stride=2),
            *block_groups,
            AveragePool(),
            # N C
            Flatten(start_dim=1, end_dim=-1),
            Linear(out_features_per_group[-1], n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)

        Return: shape (batch, n_classes)
        """
        return self.blocks(x)
