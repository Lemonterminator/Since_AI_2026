"""Portable Keras-to-PyTorch weight helpers.

These helpers take NumPy arrays that a caller extracted from a Keras
checkpoint.  They do not import TensorFlow and are not used by the runtime
model.  Keras Dense stores ``kernel`` as ``[in_features, out_features]`` while
PyTorch stores ``weight`` as ``[out_features, in_features]``.
"""

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn


def keras_dense_kernel_to_torch(kernel: np.ndarray) -> torch.Tensor:
    """Transpose a Keras Dense kernel into PyTorch linear weight layout."""
    array = np.asarray(kernel)
    if array.ndim != 2:
        raise ValueError("a Dense kernel must be a 2-D array")
    return torch.as_tensor(array.T.copy(), dtype=torch.float32)


def keras_dense_bias_to_torch(bias: np.ndarray | None) -> torch.Tensor:
    """Convert a Keras Dense bias to a PyTorch bias tensor."""
    if bias is None:
        raise ValueError("this helper needs a bias array")
    array = np.asarray(bias)
    if array.ndim != 1:
        raise ValueError("a Dense bias must be a 1-D array")
    return torch.as_tensor(array.copy(), dtype=torch.float32)


def assign_keras_dense(
    layer: nn.Linear,
    kernel: np.ndarray,
    bias: np.ndarray | None = None,
) -> nn.Linear:
    """Assign Keras Dense weights to a PyTorch ``nn.Linear`` in place."""
    weight = keras_dense_kernel_to_torch(kernel)
    if tuple(weight.shape) != tuple(layer.weight.shape):
        raise ValueError(
            f"kernel shape {tuple(weight.shape)} does not match "
            f"PyTorch weight shape {tuple(layer.weight.shape)}"
        )
    with torch.no_grad():
        layer.weight.copy_(weight)
        if bias is not None:
            bias_tensor = keras_dense_bias_to_torch(bias)
            if tuple(bias_tensor.shape) != tuple(layer.bias.shape):
                raise ValueError("bias shape does not match the PyTorch layer")
            layer.bias.copy_(bias_tensor)
    return layer


def apply_keras_weight_map(
    model: nn.Module,
    weights: Mapping[str, np.ndarray],
    mapping: Mapping[str, str],
) -> nn.Module:
    """Copy a named Keras weight map into a PyTorch module.

    ``mapping`` maps a Keras tensor name to a PyTorch parameter name.  Kernel
    entries are transposed for Linear layers. Embedding tables keep their
    original orientation. This utility is for
    parity checks only; it is intentionally independent of TensorFlow.
    """
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for keras_name, parameter_name in mapping.items():
            if keras_name not in weights:
                raise KeyError(f"missing Keras tensor {keras_name!r}")
            if parameter_name not in parameters:
                raise KeyError(f"missing PyTorch parameter {parameter_name!r}")
            array = np.asarray(weights[keras_name])
            parameter = parameters[parameter_name]
            if (parameter_name.endswith(".weight") and array.ndim == 2
                    and isinstance(model.get_submodule(parameter_name.rsplit(".", 1)[0]), nn.Linear)):
                array = array.T
            tensor = torch.as_tensor(array.copy(), dtype=parameter.dtype)
            if tuple(tensor.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"{keras_name} shape {tuple(tensor.shape)} does not match "
                    f"{parameter_name} shape {tuple(parameter.shape)}"
                )
            parameter.copy_(tensor)
    return model
