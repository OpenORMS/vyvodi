from formulae import terms as fterms
from tensorflow.python.framework import dtypes as tf_dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops.check_ops import is_numeric_tensor

from ..transforms import TRANSFORMS, Proportion, Offset
from .call_utils import CallVarsExtractor


class Call(fterms.Call):
    @property
    def var_names(self):
        """Returns the names of the variables involved in the call.

        Ensures that we are using the `formaulate` visitor to extract the
        variables, and not the `formulae` visitor.

        Returns
        -------
        result : list
            List of strings with the names of the variables in the call.
        """
        return set(CallVarsExtractor(self).get())

    def set_type(self, data_mask, env):
        """Evaluates function and determines the type of the result.

        Parameters
        -----------
        data_mask : pd.DataFrame or dict of tf.Tensor
            Mask of data from where the variables are taken.
        env : Environment
            Environment from where the values and functions are taken.

        Raises
        ------
        ValueError
            If the evaluated data is of an unknown type.
        """
        self.env = env.with_outer_namespace(TRANSFORMS)
        eval_data = self.call.eval(data_mask, self.env)

        if is_numeric_tensor(eval_data):  # already checks if it's a tensor
            self.type = 'numeric'

        if isinstance(eval_data, ops.Tensor) and eval_data.dtype == tf_dtypes.string:
            self.type = 'categoric'

        if isinstance(eval_data, ops.Tensor) and eval_data.dtype == tf_dtypes.bool:
            self.type = 'categoric'

        if isinstance(eval_data, Proportion):
            self.type = 'proportion'

        if isinstance(eval_data, Offset):
            self.type = 'offset'

        else:
            raise ValueError(
                'Call result is of unrecognized type: ({found}).'.format(
                    found=str(type(eval_data)),
                ),
            )

        self._intermediate_data = eval_data

    def set_data(self, encoding=False):
        """Finishes the evaluation of the call according to its type.

        Parameters
        ----------
        encoding : bool
            Indicates if it uses full or reduced encoding when
            the call is categoric.

        Returns
        -------
        result : pd.DataFrame or dict of tf.Tensor
            The result of the call.
        """
        pass  # TODO: implement
