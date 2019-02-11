from Graph import *
from Potential import LinearGaussianPotential
import numpy as np
from itertools import product


class KalmanFilter:
    def __init__(self, domain, transition_coeff, transition_variance, observation_coeff, observation_variance):
        self.transition_coeff = transition_coeff
        self.transition_variance = transition_variance
        self.observation_coeff = observation_coeff
        self.observation_variance = observation_variance
        self.domain = domain

    def grounded_graph(self, num_t_steps, data):
        grounded_rvs = []
        grounded_factors = []

        grounded_rvs_table = []
        for t in range(num_t_steps):
            grounded_rvs_table.append(list())

        # create all rv instance
        for t in range(num_t_steps):
            for rv_id in range(self.transition_coeff.shape[0]):
                grounding = RV(self.domain, None)
                grounded_rvs.append(grounding)
                grounded_rvs_table[t].append(grounding)

                if data[rv_id, t] != 5000:
                    # add observe node
                    observe = RV(self.domain, data[rv_id, t])
                    grounded_rvs.append(observe)

                    # add observe factors
                    obs_potential = LinearGaussianPotential(
                        self.observation_coeff[rv_id, rv_id],
                        self.observation_variance[rv_id, rv_id]
                    )
                    grounded_factors.append(
                        F(obs_potential, [grounding, observe])
                    )

        # add transition factors
        for t in range(num_t_steps - 1):
            for rv_id in range(self.transition_coeff.shape[0]):
                for rv_id_next in range(self.transition_coeff.shape[0]):
                    if self.transition_coeff[rv_id, rv_id_next] != 0:
                        transition_potential = LinearGaussianPotential(
                            self.transition_coeff[rv_id, rv_id_next],
                            self.transition_variance[rv_id, rv_id_next]
                        )
                        grounded_factors.append(
                            F(
                                transition_potential,
                                [grounded_rvs_table[t][rv_id], grounded_rvs_table[t+1][rv_id_next]])
                        )

        grounded_graph = Graph()
        grounded_graph.rvs = grounded_rvs
        grounded_graph.factors = grounded_factors
        grounded_graph.init_nb()

        return grounded_graph, grounded_rvs_table

