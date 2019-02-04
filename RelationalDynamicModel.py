from RelationalGraph import *
import numpy as np
from itertools import product


class RelationalDynamicModel:
    def __init__(self):
        self.lvs = set()
        self.atoms = set()
        self.pairwise_factors = set()
        self.observe_potentials = dict()  # atom.name: potential(x, o)
        self.transition_potentials = dict()  # atom.name: potential(x_t, x_t+1)
        self.data = dict()

    def init_nb(self):
        for atom in self.atoms:
            atom.nb = []
        for f in self.pairwise_factors:
            for atom in f.nb:
                atom.nb.append(f)

    @staticmethod
    def lvs_iter(lvs):
        table = []
        for lv in lvs:
            table.append(lv.instances)
        for combination in product(*table):
            yield dict(zip(lvs, combination))

    def grounded_dynamic_model(self, t):
        grounded_rvs = []
        grounded_factors = []

        grounded_rvs_table = []
        for i in range(t):
            grounded_rvs_table.append(dict())

        # ground all relational atoms
        for atom in self.atoms:
            for substitution in self.lvs_iter(atom.lvs):
                key = atom.key(substitution)

                for i in range(t):
                    grounding = RV(atom.domain, None)
                    grounded_rvs.append(grounding)
                    grounded_rvs_table[i][key] = grounding

                    key_t = (*key, i)  # key + time
                    if key_t in self.data:
                        # add observe node and factor
                        observe = RV(atom.domain, self.data[key_t])
                        grounded_rvs.append(observe)
                        observe_potential = self.observe_potentials[atom.name]
                        grounded_factors.append(F(observe_potential, [grounding, observe]))

        # add transition factors
        for key in grounded_rvs_table[0]:
            transition_potential = self.transition_potentials[key[0]]
            for i in range(t-1):
                rv = grounded_rvs_table[i][key]
                rv_next = grounded_rvs_table[i+1][key]
                grounded_factors.append(F(transition_potential, [rv, rv_next]))

        # add pairwise factors
        for f in self.pairwise_factors:
            # collect lvs of neighboring atom
            lvs = set()
            for atom in f.nb:
                lvs.update(atom.lvs)
            lvs = tuple(lvs)

            # enumerate all groundings and create a factor for each grounding
            for substitution in self.lvs_iter(lvs):
                if f.constrain is None or f.constrain(substitution):
                    for i in range(t):
                        # collect neighboring rv instances
                        nb = []
                        for atom in f.nb:
                            nb.append(grounded_rvs_table[i][atom.key(substitution)])
                        grounded_factors.append(F(f.potential, nb))

        grounded_graph = Graph()
        grounded_graph.rvs = grounded_rvs
        grounded_graph.factors = grounded_factors
        grounded_graph.init_nb()

        return grounded_graph, grounded_rvs_table

