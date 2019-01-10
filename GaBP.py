from Graph import *
from numpy import Inf
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import fmin
from math import sqrt

import time


class GaBP:
    # Gaussian belief propagation

    def __init__(self, g=None):
        self.g = g
        self.message = dict()

    def message_rv_to_f(self, rv, f):
        if rv.value is None:
            mu, sig = 0, 0
            for nb in rv.nb:
                if nb != f:
                    nb_mu, nb_sig = self.message[(nb, rv)]
                    mu += nb_sig ** -1 * nb_mu
                    sig += nb_sig ** -1
            sig = sig ** -1
            mu = sig * mu
            return mu, sig
        else:
            return None

    def message_f_to_rv(self, f, rv):
        # only for pairwise gaussian potential
        if rv.value is not None:
            return None

        u = f.potential.mu
        a = f.potential.sig ** -1

        rv_idx = 0
        rv_ = None
        for idx, nb in enumerate(f.nb):
            if nb == rv:
                rv_idx = idx
            else:
                rv_ = nb

        if rv_idx == 1:
            a1, a2, a3 = a[0, 0], a[0, 1], a[1, 1]
            u1, u2 = u[0], u[1]
        else:
            a1, a2, a3 = a[1, 1], a[0, 1], a[0, 0]
            u1, u2 = u[1], u[0]

        if rv_.value is None:
            m = self.message[(rv_, f)]
            a4, u3 = m[1] ** -1, m[0]
            temp = a3 * (a4 + a1) - a2 ** 2
            mu = a2 * a4 * (u1 - u3) / temp + u2
            sig = (a3 - a2 ** 2 / (a4 + a1)) ** -1
        else:
            mu = -u2 - a2 * (rv_.value - u1) / a3
            sig = a3 ** -1

        return mu, sig

    def run(self, iteration=10, log_enable=False):
        # initialize message to 1 (message from f to rv)
        for rv in self.g.rvs:
            for f in rv.nb:
                self.message[(f, rv)] = [0, 1]
                self.message[(rv, f)] = [0, 1]

        # BP iteration
        for i in range(iteration):
            print(f'iteration: {i + 1}')
            if log_enable:
                time_start = time.clock()
            # calculate messages from rv to f
            for rv in self.g.rvs:
                for f in rv.nb:
                    self.message[(rv, f)] = self.message_rv_to_f(rv, f)

            if log_enable:
                print(f'\trv to f {time.clock() - time_start}')
                time_start = time.clock()

            if i < iteration - 1:
                # calculate messages from f to rv
                for f in self.g.factors:
                    for rv in f.nb:
                        if rv.value is None:
                            self.message[(f, rv)] = self.message_f_to_rv(f, rv)

                if log_enable:
                    print(f'\tf to rv {time.clock() - time_start}')

    # def belief(self, x, rv):
    #     if rv.value is None:
    #         b = self.belief_rv(x, rv, self.sample)
    #         z = quad(lambda val: self.belief_rv(val, rv, self.sample), -Inf, Inf)[0]
    #         return b / z
    #     else:
    #         return 1 if x == rv.value else 0

    def map(self, rv):
        if rv.value is None:
            mu, sig = 0, 0
            for nb in rv.nb:
                nb_mu, nb_sig = self.message[(nb, rv)]
                mu += nb_sig ** -1 * nb_mu
                sig += nb_sig ** -1
            sig = sig ** -1
            mu = sig * mu
            return mu
        else:
            return rv.value
