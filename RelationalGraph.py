from Graph import *
import numpy as np
from itertools import product


class LV:
    # logical variable
    def __init__(self, instances):
        self.instances = instances


class Atom:
    # relational atom
    def __init__(self, domain, logical_variables, name=None):
        self.domain = domain
        self.lvs = logical_variables
        self.name = name
        self.nb = []

    def key(self, substitution):
        # substitution is a dict with format: {LV1: instance_A, LV2: instance_B, ... }
        # return key: (RelationalAtom, instance_A, instance_B, ... )
        res = [self if self.name is None else self.name]
        for lv in self.lvs:
            res.append(substitution[lv])
        return tuple(res)


class RelationalGraph:
    def __init__(self):
        self.lvs = set()
        self.rvs = set()
        self.factors = set()
        self.data = dict()  # format: key=(RelationalAtom, LV1_instance, LV2_instance, ... ) value=True or 0.01 etc.

    def init_nb(self):
        for rv in self.rvs:
            rv.nb = []
        for f in self.factors:
            for rv in f.nb:
                rv.nb.append(f)

    @staticmethod
    def lvs_iter(lvs):
        table = []
        for lv in lvs:
            table.append(lv.instances)
        for combination in product(*table):
            yield dict(zip(lvs, combination))

    def grounded_graph(self):
        grounded_rvs = []
        grounded_factors = []

        grounded_rvs_table = dict()

        # ground all relational atoms
        for rv in self.rvs:
            if type(rv) is Atom:
                for substitution in self.lvs_iter(rv.lvs):
                    key = rv.key(substitution)
                    value = self.data[key] if key in self.data else None
                    grounding = RV(rv.domain, value)
                    grounded_rvs_table[key] = grounding
                    grounded_rvs.append(grounding)
            else:
                grounded_rvs.append(rv)

        # add factors
        for f in self.factors:
            # collect lvs of neighboring atom
            lvs = set()
            for rv in f.nb:
                if type(rv) is Atom:
                    lvs.update(rv.lvs)
            lvs = tuple(lvs)

            # enumerate all groundings and create a factor for each grounding
            for substitution in self.lvs_iter(lvs):
                # collect neighboring rv instances
                nb = []
                for rv in f.nb:
                    if type(rv) is Atom:
                        nb.append(grounded_rvs_table[rv.key(substitution)])
                    else:
                        nb.append(rv)
                grounded_factors.append(F(f.potential, nb))

        grounded_graph = Graph()
        grounded_graph.rvs = grounded_rvs
        grounded_graph.factors = grounded_factors
        grounded_graph.init_nb()

        return grounded_graph, grounded_rvs_table

