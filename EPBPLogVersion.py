from Graph import *
import numpy as np
from numpy import Inf, exp
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import fminbound
from statistics import mean
from math import sqrt, log, e, erf
from itertools import product

import time


class EPBP:
    # Expectation particle belief propagation

    var_threshold = 5
    max_log_value = 700

    def __init__(self, g=None, n=50, proposal_approximation='EP'):
        self.g = g
        self.n = n
        self.proposal_approximation = proposal_approximation
        self.message = dict()  # log message, message in log space
        self.sample = dict()
        self.q = dict()
        self.eta_message = dict()

    @staticmethod
    def gaussian_product(*gaussian):
        # input a list of gaussian's mean and variance
        # output the product distribution's mean and variance
        mu, sig = 0, 0
        for g in gaussian:
            mu_, sig_ = g
            sig += sig_ ** -1
            mu += sig_ ** -1 * mu_
        sig = sig ** -1
        mu = sig * mu
        return mu, sig

    @staticmethod
    def gaussian_division(a, b):
        sig = a[1] * b[1] / (b[1] - a[1])
        mu = (a[0] * (b[1] + sig) - b[0] * sig) / b[1]
        return mu, sig

    @staticmethod
    def norm_pdf(x, mu, sig):
        u = (x - mu) / sig
        y = exp(-u * u * 0.5) / (2.506628274631 * sig)
        return y

    @staticmethod
    def norm_cdf(x, mu, sig):
        u = (x - mu) / sqrt(sig * 2)
        y = (1 + erf(u)) * 0.5
        return y

    def generate_sample(self):
        sample = dict()
        for rv in self.g.rvs:
            if rv.value is None:
                if rv.domain.continuous:
                    sample[rv] = np.clip(norm(self.q[rv][0], sqrt(self.q[rv][1])).rvs(self.n),
                                         a_min=rv.domain.values[0], a_max=rv.domain.values[1])
                else:
                    sample[rv] = rv.domain.values
        return sample

    def initial_proposal(self):
        for rv in self.g.rvs:
            if rv.value is None:
                self.q[rv] = (0, 5)

                count = len(rv.nb)  # count the number of incoming messages
                site = (0, 5 * count)

                for f in rv.nb:
                    self.eta_message[(f, rv)] = site

    def update_proposal(self):
        for rv in self.g.rvs:
            if rv.value is None and rv.domain.continuous:
                eta = list()
                min_sig = len(rv.nb) * self.var_threshold
                for f in rv.nb:
                    if self.proposal_approximation == 'EP':
                        mu, sig = self.eta_approximation(f, rv)
                    else:
                        mu, sig = self.eta_approximation_simple(f, rv)
                    if 0 < sig < Inf:
                        sig = max(sig, min_sig)
                        self.eta_message[(f, rv)] = (mu, sig)
                    else:
                        mu, sig = self.eta_message[(f, rv)]
                    eta.append((mu, sig))
                # old_q = self.q[rv]
                self.q[rv] = self.gaussian_product(*eta)
                # print(f'{old_q} -> {self.q[rv]}')

    def eta_approximation_simple(self, f, rv):
        # approximate the message directly
        weight = []
        mu = 0
        sig = 0

        for x in rv.domain.integral_points:
            weight.append(e ** self.message[(f, rv)][x])

        z = sum(weight)

        for w, x in zip(weight, rv.domain.integral_points):
            mu += w * x
            sig += w * x ** 2

        mu = mu / z
        sig = sig / z - mu ** 2

        return mu, sig

    def eta_approximation(self, f, rv):
        # compute the cavity distribution
        a, b = self.q[rv], self.eta_message[(f, rv)]

        if a[1] >= b[1]:
            return self.eta_approximation_simple(f, rv)

        cavity = self.gaussian_division(a, b)

        # compute the momentum of tilted distribution
        weight = []
        mu = 0
        sig = 0

        param = (cavity[0], sqrt(cavity[1]))
        for x in rv.domain.integral_points:
            weight.append(e ** self.message[(f, rv)][x] * self.norm_pdf(x, *param))

        z = sum(weight)

        for w, x in zip(weight, rv.domain.integral_points):
            mu += w * x
            sig += w * x ** 2

        mu = mu / z
        sig = sig / z - mu ** 2

        # if sig >= cavity[1]:
        #     return self.eta_approximation_simple(f, rv)

        # approximate eta
        return self.gaussian_division((mu, sig), cavity)

    def important_weight(self, x, rv):
        if rv.value is None:
            if x == rv.domain.values[0] or x == rv.domain.values[1]:
                return 1e-200
            else:
                return 1 / max(self.norm_pdf(x, self.q[rv][0], sqrt(self.q[rv][1])), 1e-200)
        else:
            return 1

    def message_rv_to_f(self, x, rv, f):
        # the incoming message on x must be calculated before calculating the out going message
        # that means x should not be re-sampled
        # (before calculating out going message and after calculating incoming message)
        if rv.value is None:
            res = 0
            for nb in rv.nb:
                if nb != f:
                    res += self.message[(nb, rv)][x]
            return res + log(self.important_weight(x, rv))

    def message_f_to_rv(self, x, f, rv, sample):
        # sample is a set of sample points of neighbouring rvs
        # incoming message should be calculated before this process
        res = 0
        param = []
        for nb in f.nb:
            if nb == rv:
                param.append((x,))
            elif nb.value is None:
                param.append(sample[nb])
            else:
                param.append((nb.value,))
        for x_join in product(*param):
            m = 0
            for idx, nb in enumerate(f.nb):
                if nb != rv and nb.value is None:
                    m += self.message[(nb, f)][x_join[idx]]
            res += f.potential.get(x_join) * e ** m
        return log(res) if res > 0 else -700

    def belief_rv(self, x, rv, sample):
        # sample is a set of sample points of neighbouring rvs
        # incoming message (from neighbouring rvs to neighbouring f) should be calculated before this process
        res = 0
        for f in rv.nb:
            res += self.message_f_to_rv(x, f, rv, sample)
        return res

    def log_message_balance(self, message):
        values = message.values()
        mean_m = mean(values)
        max_m = max(values)
        if max_m - mean_m > self.max_log_value:
            shift = max_m - self.max_log_value
        else:
            shift = mean_m
        for k, v in message.items():
            message[k] = v - shift

        return shift

    @staticmethod
    def message_normalization(message):
        z = 0
        for k, v in message.items():
            z = z + v
        for k, v in message.items():
            message[k] = v / z

    def run(self, iteration=10, log_enable=False):
        # initialize proposal
        self.initial_proposal()

        # poll sample from the initial distribution
        self.sample = self.generate_sample()

        # initialize log message to 0 (message from f to rv)
        for rv in self.g.rvs:
            if rv.value is None:
                for f in rv.nb:
                    m = {k: 0 for k in self.sample[rv]}
                    if rv.domain.continuous:
                        eta_m = {k: 0 for k in rv.domain.integral_points}
                        self.message[(f, rv)] = {**m, **eta_m}
                    else:
                        self.message[(f, rv)] = m
                    self.message[(rv, f)] = m

        # BP iteration
        for i in range(iteration):
            print(f'iteration: {i + 1}')
            if log_enable:
                time_start = time.clock()
            # calculate messages from rv to f
            for rv in self.g.rvs:
                if rv.value is None:
                    for f in rv.nb:
                        # compute the message for each sample point
                        m = dict()
                        for point in self.sample[rv]:
                            m[point] = self.message_rv_to_f(point, rv, f)
                        self.log_message_balance(m)
                        self.message[(rv, f)] = m

            if log_enable:
                print(f'\trv to f {time.clock() - time_start}')
                time_start = time.clock()

            if i < iteration - 1:
                # update proposal
                self.update_proposal()
                if log_enable:
                    print(f'\tproposal {time.clock() - time_start}')

                # poll new sample
                old_sample = self.sample
                self.sample = self.generate_sample()

                # calculate messages from f to rv
                for f in self.g.factors:
                    for rv in f.nb:
                        if rv.value is None:
                            # compute the message for each sample point
                            m = dict()
                            for point in self.sample[rv]:
                                m[point] = self.message_f_to_rv(point, f, rv, old_sample)
                            if rv.domain.continuous:
                                for point in rv.domain.integral_points:
                                    m[point] = self.message_f_to_rv(point, f, rv, old_sample)
                            self.message[(f, rv)] = m

                if log_enable:
                    print(f'\tf to rv {time.clock() - time_start}')
                    time_start = time.clock()

    def log_area(self, f, a, b, n, shift=None):
        res = 0
        x = linspace(a, b, n)
        d = x[1] - x[0]
        y = dict()
        for i, v in enumerate(x):
            y[i] = f(v)
        if shift is None:
            shift = self.log_message_balance(y)
        else:
            for k, v in y.items():
                y[k] = v - shift
        prev = e ** y[0]
        for i in range(1, n):
            current = e ** y[i]
            res += (prev + current) * d
            prev = current
        return res * 0.5, shift

    def belief(self, x, rv):
        if rv.value is None:
            # z = quad(
            #     lambda val: e ** self.belief_rv(val, rv, self.sample),
            #     rv.domain.values[0], rv.domain.values[1]
            # )[0]
            z, shift = self.log_area(
                lambda val: self.belief_rv(val, rv, self.sample),
                rv.domain.values[0], rv.domain.values[1],
                20
            )

            b = e ** (self.belief_rv(x, rv, self.sample) - shift)

            return b / z
        else:
            return 1 if x == rv.value else 0

    def probability(self, a, b, rv):
        # only for continuous hidden variable
        if rv.value is None:
            if rv.domain.continuous:
                z, shift = self.log_area(
                    lambda val: self.belief_rv(val, rv, self.sample),
                    rv.domain.values[0], rv.domain.values[1],
                    20
                )

                b, _ = self.log_area(
                    lambda val: self.belief_rv(val, rv, self.sample),
                    a, b,
                    5,
                    shift
                )

                return b / z

        return None

    def map(self, rv):
        if rv.value is None:
            res = fminbound(
                lambda val: -self.belief_rv(val, rv, self.sample),
                rv.domain.values[0], rv.domain.values[1],
                disp=False
            )
            return res
        else:
            return rv.value
