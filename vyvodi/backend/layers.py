"""Submodule for layers.

This submodule contains the methods for interpreting
Wilkonsin's formulas and converting them to the
corresponding TensorFlow Layers.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from formulae import model_description


class Layer(object):
    """TODO: Docstring for __call__.

    Attributes:
        model: tbd
        response: tbd
        common_terms: tbd
        group_terms: tbd
    """

    def __init__(self, model):
        """TODO: Docstring for __init__.

        Parameters:
            model: TODO
        """
        self.model = model
        self.response = None
        self.common_terms = None
        self.group_terms = None

        if self.model.response:
            self.response = self.model.response

        if self.model.common_terms:
            self.common_terms = self.model.common_terms

        if self.model.group_terms:
            self.group_terms = self.model.group_terms


class ResponseLayer(object):
    """TODO: Docstring for __call__.

    Parameters:
        model: TODO
    """

    def __init__(self, term):
        """TODO: Docstring for __init__.

        Parameters:
            term: TODO
        """
        self.term = term
        self.name = None
        self.kind = None  # either numeric or categorical
        self.baseline = None  # Not None for non-binary categorical
        self.success = None  # Not None if binary categorical
        self.levels = None  # Not None for categorical
        self.binary = None  # Not None for categorical (bool)
    
    def _evaluate(self):
        """TODO: Docstring for _evaluate.
        """
        self.name = self.term.term.name
        self.kind = self.term.term.metadata['kind']

        if self.kind == 'categoric':
            self.binary = None  # TODO: need to figure out how to do this in tf
            self.levels = self.term.term.metadata['levels']
            if self.binary:
                self.success = self.term.term.metadata['reference']
            else:
                self.baseline = self.term.term.metadata['reference']


class CommonTermsLayer(object):
    """TODO: Docstring for __call__.

    Parameters:
        x: TODO

    Attributes:
        terms: TODO
    """

    def __init__(self, terms):
        """TODO: Docstring for __init__.

        Parameters:
            terms: TODO
        """
        self.terms = terms
