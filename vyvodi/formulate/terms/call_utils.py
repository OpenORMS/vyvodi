import formulae.terms as fterms


class CallVarsExtractor(fterms.CallVarsExtractor):
    """Visitor that extracts variables names from a model expression.

    Might be unnecessary, given that it is just a stub, but it is
    implemented as a separate class to avoid confusion with the
    `CallVarsExtractor` class in `formulae` and in case we want to
    implement a different visitor for the model expressions in the
    future.
    """
    def __init__(self, expr):
        super().__init__(expr)
