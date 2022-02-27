from tensorflow.python.framework import dtypes, ops
from tensorflow.python.ops.check_ops import NUMERIC_TYPES


def is_numeric_tensor(tensor):
    """Checks if the tensor is numeric.

    Parameters
    ----------
    tensor : tf.Tensor
        Tensor to be checked.

    Returns
    -------
    result : bool
        True if the tensor is numeric, False otherwise.
    """
    if isinstance(tensor, ops.Tensor):
        return tensor.dtype in NUMERIC_TYPES + [dtypes.bool]

    return False


def is_string_tensor(tensor):
    """Checks if the tensor is a string.

    Parameters
    ----------
    tensor : tf.Tensor
        Tensor to be checked.

    Returns
    -------
    result : bool
        True if the tensor is a string, False otherwise.
    """
    if isinstance(tensor, ops.Tensor):
        return tensor.dtype == dtypes.string

    return False
