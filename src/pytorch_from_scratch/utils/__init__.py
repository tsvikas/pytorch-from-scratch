import torch


class TensorDataset:
    def __init__(self, *tensors: torch.Tensor):
        """Validate the sizes and store the tensors in a field named `tensors`."""
        if tensors:
            shape = tensors[0].shape
            for tensor in tensors:
                if tensor.shape[0] != shape[0]:
                    raise ValueError(
                        "all tensors should have the same length in the first dimension"
                    )
        self.tensors = tensors

    def __getitem__(self, index: int | slice) -> tuple[torch.Tensor, ...]:
        """Return a tuple of length len(self.tensors) with the index applied to each."""
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        """Return the size in the first dimension, common to all the tensors."""
        return self.tensors[0].shape[0]
