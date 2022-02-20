from tensorflow.python.framework import dtypes, ops
from tensorflow.python.ops.check_ops import NUMERIC_TYPES

CATEGORIC_TYPES = frozenset((dtypes.bool, dtypes.string))


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
        return tensor.dtype in NUMERIC_TYPES

    return False


def is_categoric_tensor(tensor):
    """Checks if the tensor is categoric.

    Parameters
    ----------
    tensor : tf.Tensor
        Tensor to be checked.

    Returns
    -------
    result : bool
        True if the tensor is categoric, False otherwise.
    """
    if isinstance(tensor, ops.Tensor):
        return tensor.dtype in CATEGORIC_TYPES

    return False
