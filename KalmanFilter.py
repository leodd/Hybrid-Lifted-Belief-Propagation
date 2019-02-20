from Graph import *
import numpy as np
from Potential import LinearGaussianPotential, XYPotential, X2Potential
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

        n = self.transition_coeff.shape[0]

        # create all rv instance
        for t in range(num_t_steps):
            for x in range(self.transition_coeff.shape[0]):
                if t == 0:
                    grounding = RV(self.domain, data[x, 0])
                    grounded_rvs.append(grounding)
                    grounded_rvs_table[t].append(grounding)
                else:
                    grounding = RV(self.domain, None)
                    grounded_rvs.append(grounding)
                    grounded_rvs_table[t].append(grounding)

                    if data[x, t] != 5000:
                        # add observe node
                        observe = RV(self.domain, data[x, t])
                        grounded_rvs.append(observe)

                        # add observe factors
                        obs_potential = LinearGaussianPotential(
                            self.observation_coeff[x, x],
                            self.observation_variance
                        )
                        grounded_factors.append(
                            F(obs_potential, [grounding, observe])
                        )

        # compute decompose transition potential parameters
        xy_coeff = self.transition_coeff
        xx_coeff = np.zeros((n, n))
        for y in range(n):
            xx_coeff += np.outer(xy_coeff[:, y], xy_coeff[:, y])

        # add transition factors
        for t in range(num_t_steps - 1):
            for x in range(n):
                # add node factor
                if t > 0:
                    if xx_coeff[x, x] != 0:
                        grounded_factors.append(
                            F(
                                X2Potential(xx_coeff[x, x], self.transition_variance),
                                [grounded_rvs_table[t][x]]
                            )
                        )

                for y in range(n):
                    # add xy factor
                    if xy_coeff[x, y] != 0:
                        grounded_factors.append(
                            F(
                                XYPotential(-2 * xy_coeff[x, y], self.transition_variance),
                                [grounded_rvs_table[t][x], grounded_rvs_table[t + 1][y]]
                            )
                        )

                    # add xx factor
                    if t > 0 and x < y:
                        if xx_coeff[x, y] != 0:
                            grounded_factors.append(
                                F(
                                    XYPotential(2 * xx_coeff[x, y], self.transition_variance),
                                    [grounded_rvs_table[t][x], grounded_rvs_table[t][y]]
                                )
                            )

        for t in range(1, num_t_steps):
            for x in range(n):
                grounded_factors.append(
                    F(
                        X2Potential(1, self.transition_variance),
                        [grounded_rvs_table[t][x]]
                    )
                )

        grounded_graph = Graph()
        grounded_graph.rvs = grounded_rvs
        grounded_graph.factors = grounded_factors
        grounded_graph.init_nb()

        return grounded_graph, grounded_rvs_table

    # def grounded_graph(self, num_t_steps, data):
    #     grounded_rvs = []
    #     grounded_factors = []
    #
    #     grounded_rvs_table = []
    #     for t in range(num_t_steps):
    #         grounded_rvs_table.append(list())
    #
    #     # create all rv instance
    #     for t in range(num_t_steps):
    #         for rv_id in range(self.transition_coeff.shape[0]):
    #             if t == 0:
    #                 grounding = RV(self.domain, data[rv_id, 0])
    #                 grounded_rvs.append(grounding)
    #                 grounded_rvs_table[t].append(grounding)
    #             else:
    #                 grounding = RV(self.domain, None)
    #                 grounded_rvs.append(grounding)
    #                 grounded_rvs_table[t].append(grounding)
    #
    #                 if data[rv_id, t] != 5000:
    #                     # add observe node
    #                     observe = RV(self.domain, data[rv_id, t])
    #                     grounded_rvs.append(observe)
    #
    #                     # add observe factors
    #                     obs_potential = LinearGaussianPotential(
    #                         self.observation_coeff[rv_id, rv_id],
    #                         self.observation_variance
    #                     )
    #                     grounded_factors.append(
    #                         F(obs_potential, [grounding, observe])
    #                     )
    #
    #     # add transition factors
    #     for t in range(num_t_steps - 1):
    #         for rv_id in range(self.transition_coeff.shape[0]):
    #             for rv_id_next in range(self.transition_coeff.shape[0]):
    #                 if self.transition_coeff[rv_id, rv_id_next] != 0:
    #                     transition_potential = LinearGaussianPotential(
    #                         self.transition_coeff[rv_id, rv_id_next],
    #                         self.transition_variance
    #                     )
    #                     grounded_factors.append(
    #                         F(
    #                             transition_potential,
    #                             [grounded_rvs_table[t][rv_id], grounded_rvs_table[t+1][rv_id_next]])
    #                     )
    #
    #     grounded_graph = Graph()
    #     grounded_graph.rvs = grounded_rvs
    #     grounded_graph.factors = grounded_factors
    #     grounded_graph.init_nb()
    #
    #     return grounded_graph, grounded_rvs_table
