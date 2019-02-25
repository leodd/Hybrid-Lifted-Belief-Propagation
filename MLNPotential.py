from Potential import Potential
from math import e
import numpy as np


def and_op(x, y):
    return x * y


def or_op(x, y):
    return x + y - x * y


def neg_op(x):
    return 1 - x


def imp_op(x, y):
    return or_op(1 - x, y)


def bic_op(x, y):
    return imp_op(x, y) * imp_op(y, x)


def eq_op(x, y):
    return -(x - y) ** 2


class MLNPotential(Potential):
    def __init__(self, formula, w=1):
        Potential.__init__(self, symmetric=False)
        self.formula = formula
        self.w = w

    def get(self, parameters):
        return e ** (self.formula(parameters) * self.w)


class MLNNodePotential(Potential):
    def __init__(self):
        Potential.__init__(self, symmetric=False)

    def get(self, parameters):
        return e ** (0 if 0 <= parameters[0] <= 1 else -10)
