from Graph import *
from numpy import Inf, linspace
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import fmin
from statistics import mean
from math import sqrt, log, e
from itertools import product

import time


class EPBP:
    # Expectation particle belief propagation

    var_threshold = 0.2
    max_log_value = 700

    def __init__(self, g=None, n=50):
        self.g = g
        self.n = n
        self.message = dict()  # log message, message in log space
        self.sample = dict()
        self.q = dict()
        self.custom_initial_proposal = self.initial_proposal

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

    def generate_sample(self):
        sample = dict()
        for rv in self.g.rvs:
            if rv.value is None:
                sample[rv] = norm(self.q[rv][0], sqrt(self.q[rv][1])).rvs(self.n)
            else:
                sample[rv] = (rv.value,)
        return sample

    def initial_proposal(self):
        for rv in self.g.rvs:
            if rv.value is None:
                self.q[rv] = (0, 2)
            else:
                self.q[rv] = None

    def update_proposal(self):
        for rv in self.g.rvs:
            if rv.value is None:
                eta = list()
                for f in rv.nb:
                    eta.append(self.eta_message_f_to_rv(f, rv))
                new_q = self.gaussian_product(*eta)
                if new_q[1] < self.var_threshold:
                    new_q = (new_q[0], self.q[rv][1])
                self.q[rv] = new_q
            else:
                self.q[rv] = None

    def important_weight(self, x, rv):
        if rv.value is None:
            return 1 / norm(self.q[rv][0], sqrt(self.q[rv][1])).pdf(x)
        else:
            return 1

    def eta_message_f_to_rv(self, f, rv):
        # eta_message is the gaussian approximation of the particle message
        # the return is a tuple of mean and variance
        weight = []
        mu = 0
        var = 0

        for x in rv.domain.integral_points:
            weight.append(e ** self.message[(f, rv)][x])

        z = sum(weight)

        for w, x in zip(weight, rv.domain.integral_points):
            mu += w * x
            var += w * x ** 2

        mu = mu / z
        var = var / z - mu ** 2

        return mu, (var if var > self.var_threshold else self.var_threshold)

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
        else:
            return 0

    def message_f_to_rv(self, x, f, rv, sample):
        # sample is a set of sample points of neighbouring rvs
        # incoming message should be calculated before this process
        res = 0
        param = []
        for nb in f.nb:
            if nb == rv:
                param.append((x,))
            else:
                param.append(sample[nb])
        for x_join in product(*param):
            m = 0
            for i in range(len(f.nb)):
                if f.nb[i] != rv:
                    m += self.message[(f.nb[i], f)][x_join[i]]
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
        shift = mean(values)
        max_m = max(values) - shift
        if max_m > self.max_log_value:
            shift = max_m - self.max_log_value
        for k, v in message.items():
            message[k] = v - shift

    @staticmethod
    def message_normalization(message):
        z = 0
        for k, v in message.items():
            z = z + v
        for k, v in message.items():
            message[k] = v / z

    def run(self, iteration=10, log_enable=False):
        # initialize proposal
        self.custom_initial_proposal()

        # poll sample from the initial distribution
        self.sample = self.generate_sample()

        # initialize log message to 0 (message from f to rv)
        for rv in self.g.rvs:
            for f in rv.nb:
                m = {k: 0 for k in self.sample[rv]}
                eta_m = {k: 0 for k in rv.domain.integral_points}
                self.message[(f, rv)] = {**m, **eta_m}
                self.message[(rv, f)] = m

        # BP iteration
        for i in range(iteration):
            print(f'iteration: {i + 1}')
            if log_enable:
                time_start = time.clock()
            # calculate messages from rv to f
            for rv in self.g.rvs:
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
                    time_start = time.clock()

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
                            # compute the eta message for each integral point
                            eta_m = dict()
                            for point in rv.domain.integral_points:
                                eta_m[point] = self.message_f_to_rv(point, f, rv, old_sample)
                            self.log_message_balance(eta_m)
                            self.message[(f, rv)] = {**m, **eta_m}

                if log_enable:
                    print(f'\tf to rv {time.clock() - time_start}')

    def belief(self, x, rv):
        if rv.value is None:
            b = e ** self.belief_rv(x, rv, self.sample)
            z = quad(lambda val: e ** self.belief_rv(val, rv, self.sample), -Inf, Inf)[0]
            return b / z
        else:
            return 1 if x == rv.value else 0

    def map(self, rv):
        if rv.value is None:
            res = fmin(lambda val: -self.belief_rv(val, rv, self.sample), 0, disp=False)[0]
            return res
        else:
            return rv.value
