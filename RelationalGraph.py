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


class ParamF:
    # parametric factor
    def __init__(self, potential, nb=None, constrain=None):
        self.potential = potential
        self.constrain = constrain
        if nb is None:
            self.nb = []
        else:
            self.nb = nb


class RelationalGraph:
    def __init__(self):
        self.lvs = set()
        self.atoms = set()
        self.param_factors = set()
        self.data = dict()  # format: key=(RelationalAtom, LV1_instance, LV2_instance, ... ) value=True or 0.01 etc.

    def init_nb(self):
        for atom in self.atoms:
            atom.nb = []
        for f in self.param_factors:
            for atom in f.nb:
                atom.nb.append(f)

    @staticmethod
    def lvs_iter(lvs):
        table = []
        for lv in lvs:
            table.append(lv.instances)
        for combination in product(*table):
            yield dict(zip(lvs, combination))

    def key_list(self):
        res = list()
        for atom in self.atoms:
            for substitution in self.lvs_iter(atom.lvs):
                key = atom.key(substitution)
                res.append(key)

        return res

    def grounded_graph(self):
        grounded_factors = set()
        grounded_rvs_table = dict()

        # ground all relational atoms
        for atom in self.atoms:
            for substitution in self.lvs_iter(atom.lvs):
                key = atom.key(substitution)
                value = self.data[key] if key in self.data else None
                grounding = RV(atom.domain, value)
                grounded_rvs_table[key] = grounding

        # add factors
        for param_f in self.param_factors:
            # collect lvs of neighboring atom
            lvs = set()
            for atom in param_f.nb:
                lvs.update(atom.lvs)
            lvs = tuple(lvs)

            # enumerate all groundings and create a factor for each grounding
            for substitution in self.lvs_iter(lvs):
                if param_f.constrain is None or param_f.constrain(substitution):
                    # collect neighboring rv instances
                    nb = []
                    for atom in param_f.nb:
                        nb.append(grounded_rvs_table[atom.key(substitution)])
                    grounded_factors.add(F(param_f.potential, nb))

        grounded_graph = Graph()
        grounded_graph.rvs = set(grounded_rvs_table.values())
        grounded_graph.factors = grounded_factors
        grounded_graph.init_nb()

        # remove unconnected rvs
        keys = tuple(grounded_rvs_table.keys())
        for key in keys:
            if len(grounded_rvs_table[key].nb) == 0:
                del grounded_rvs_table[key]

        grounded_graph.rvs = set(grounded_rvs_table.values())

        return grounded_graph, grounded_rvs_table

