from formulae.terms import Model


class DesignNetwork(object):
    def __init__(self, model, data, env):
        self.model = model
        self.data = data
        self.env = env

        if self.model.common_terms:
            self.common = CommonEffectsInput(Model(*self.model.common_terms))
            self.common._evaluate(data, env)

        if self.model.group_terms:
            self.group = GroupEffectsInput(self.model.group_terms)
            self.group._evaluate(data, env)


class CommonEffectsInput(object):
    def __init__(self, model):
        self.model = model
        self.data = None
        self.env = None
        self.terms_info = None
        self.evaluated = False

    def _evaluate(self, data, env):
        self.data = data
        self.env = env
        self.terms_info = {}

        self.model.set_types(data, env)  # struggles with non-numpy arrays

        for term in self.model.terms:
            self.terms_info[term.name] = term.metadata

        self.evaluated = True


class GroupEffectsInput(object):
    pass
