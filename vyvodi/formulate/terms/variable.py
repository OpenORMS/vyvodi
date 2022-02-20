import tensorflow as tf
from formulae.terms.variable import Variable as FormulaeVariable

from vyvodi.formulate.terms import type_checker

tnp = tf.experimental.numpy


class Variable(FormulaeVariable):  # noqa: WPS110
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
        elif type_checker.is_categoric(eval_data):
            self.kind = 'categoric'
        else:
            raise ValueError('...')  # TODO: raise error
        self._intermediate_data = eval_data

    def _eval_numeric(self, x):  # noqa: WPS111
        if isinstance(x, tf.Tensor):
            value = tnp.atleast_2d(x)
            if x.shape[0] == 1 and x.shape[1] > 1:
                value = value.T

            results = {'value': value, 'type': 'numeric'}
        else:
            results = super()._eval_numeric(x)
            results['value'] = tf.convert_to_tensor(results['value'])

        return results
