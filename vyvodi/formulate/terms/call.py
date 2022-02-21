from formulae.terms.call import Call as FormulaeCall
from formulae.terms.call_utils import CallVarsExtractor
from pandas import DataFrame

from vyvodi.formulate.terms import type_checker
from vyvodi.formulate.transforms import TRANSFORMS, Offset, Proportion


class Call(FormulaeCall):
    """Representation of a call in a model Term.

    This class and ``Variable`` ar the atomic components of a model term.
    """

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

    def set_type(self, data_mask, env):  # noqa: WPS615
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
        if isinstance(data_mask, DataFrame):
            super().set_type(data_mask, env)

        else:
            self.env = env.with_outer_namespace(TRANSFORMS)
            eval_data = self.call.eval(data_mask, self.env)

            if type_checker.is_numeric(eval_data):
                self.type = 'numeric'

            if type_checker.is_categoric(eval_data):
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
