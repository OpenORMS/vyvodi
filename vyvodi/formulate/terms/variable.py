import tensorflow as tf
from formulae.terms.variable import Variable as FormulaeVariable

from vyvodi.formulate.terms import type_checker

tnp = tf.experimental.numpy


class Variable(FormulaeVariable):
    """Representation of a variable in a model Term.

    This class and ``Call`` ar the atomic components of a model term.
    """

    def set_type(self, data_mask):
        """Determines the type of the variable.

        Parameters
        ----------
        data_mask : pd.DataFrame or dict of tf.Tensor
            The data to be used to determine the type.

        Raises
        ------
        ValueError
            If the type of the variable cannot be determined
            or if the variable is numeric and has a level.
        """
        eval_data = data_mask[self.name]
        if type_checker.is_numeric(eval_data):
            self.kind = 'numeric'
            if self.level is not None:
                raise ValueError('...')  # TODO: raise error
        if type_checker.is_string_tensor(eval_data):
            self.kind = 'categoric'

        if type_checker.is_bool_tensor(eval_data):
            self.kind = 'categoric'
        else:
            raise ValueError('...')  # TODO: raise error
        self._intermediate_data = eval_data

    def _eval_numeric(self, x):
        if isinstance(x, tf.Tensor):
            value = tnp.atleast_2d(x)  # Ensures concatenation works.
            if x.shape[0] == 1 and x.shape[1] > 1:
                value = tf.transpose(value)
            kind = 'numeric'
        else:
            # Use the base class implementation, then convert to a tensor.
            value, kind = super()._eval_numeric(x).values()
            value = tf.convert_to_tensor(value)
            # TODO: make the dtype more explicit.

        return {'value': value, 'type': kind}

    def _eval_categoric(self, x, encoding):
        if x.dtype == tf.bool:
            x = tf.cast(x, tf.float32)
            value, _ = self._eval_numeric(x).values()
            levels = None
            reference = False
        elif x.dtype == tf.string:
            levels, value = tf.unique(x)
            reference = levels[-1]
            if encoding:
                value = tf.one_hot(value, levels.shape[0])
                encoding = 'full'
            else:
                value = tf.one_hot(value, levels.shape[0] - 1)
                encoding = 'reduced'

        return {
            'value': value,
            'type': 'categoric',
            'levels': levels,
            'reference': reference,
            'encoding': encoding,
        }
