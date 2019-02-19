from abc import ABC, abstractmethod
from numpy import linspace


class Domain:
    def __init__(self, values, continuous=False, integral_points=None):
        self.values = tuple(values)
        self.continuous = continuous
        if continuous:
            if integral_points is None:
                self.integral_points = linspace(values[0], values[1], 30)
            else:
                self.integral_points = integral_points

    # def __hash__(self):
    #     return hash((self.values, self.continuous))
    #
    # def __eq__(self, other):
    #     return (
    #         self.__class__ == other.__class__ and
    #         self.values == other.values and
    #         self.continuous == other.continuous
    #     )


class Potential(ABC):
    def __init__(self, symmetric=False):
        self.symmetric = symmetric

    @abstractmethod
    def get(self, parameters):
        pass


class RV:
    def __init__(self, domain, value=None):
        self.domain = domain
        self.value = value
        self.nb = []


class F:
    def __init__(self, potential, nb=None):
        self.potential = potential
        if nb is None:
            self.nb = []
        else:
            self.nb = nb


class Graph:
    def __init__(self):
        self.rvs = set()
        self.factors = set()

    def init_nb(self):
        for rv in self.rvs:
            rv.nb = []
        for f in self.factors:
            for rv in f.nb:
                rv.nb.append(f)
