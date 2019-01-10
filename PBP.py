from Graph import *
from numpy import Inf, linspace
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import fmin
from math import sqrt
from itertools import product

import time


class PBP:
    # particle belief propagation

    var_threshold = 0.2

    def __init__(self, g=None, n=50):
        self.g = g
        self.n = n
        self.message = dict()
        self.sample = dict()
        self.old_sample = dict()
        self.q = dict()
        self.custom_initial_proposal = self.initial_proposal
        self.integral_points = linspace(-30, 130, 30)

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

    def important_weight(self, x, rv):
        if rv.value is None:
            return 1 / (self.n * norm(self.q[rv][0], sqrt(self.q[rv][1])).pdf(x))
        else:
            return 1

    def message_rv_to_f(self, x, rv, f):
        # the incoming message on x must be calculated before calculating the out going message
        # that means x should not be re-sampled
        # (before calculating out going message and after calculating incoming message)
        if rv.value is None:
            res = 1
            for nb in rv.nb:
                if nb != f:
                    res = res * self.message[(nb, rv)][x]
            return res * self.important_weight(x, rv)
        else:
            return 1

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
            m = 1
            for i in range(len(f.nb)):
                if f.nb[i] != rv:
                    m = m * self.message[(f.nb[i], f)][x_join[i]]
            res = res + f.potential.get(x_join) * m
        return res

    @staticmethod
    def message_normalization(message):
        z = 0
        for k, v in message.items():
            z = z + v
        for k, v in message.items():
            message[k] = v / z

    def run(self, iteration=10):
        # initialize proposal
        self.custom_initial_proposal()

        # poll sample from the initial distribution
        self.sample = self.generate_sample()

        # initialize message to 1 (message from f to rv)
        for f in self.g.factors:
            for rv in f.nb:
                self.message[(f, rv)] = {k: 1 for k in self.sample[rv]}

        for rv in self.g.rvs:
            for f in rv.nb:
                self.message[(rv, f)] = dict()

        # BP iteration
        for i in range(iteration):
            print(f'iteration: {i + 1}')
            time_start = time.clock()
            # calculate messages from rv to f
            for rv in self.g.rvs:
                for f in rv.nb:
                    # compute the message for each sample point
                    m = dict()
                    for point in self.sample[rv]:
                        m[point] = self.message_rv_to_f(point, rv, f)
                    self.message[(rv, f)] = m

            # poll new sample
            self.old_sample = self.sample
            self.sample = self.generate_sample()

            # calculate messages from f to rv
            for f in self.g.factors:
                for rv in f.nb:
                    # compute the message for each sample point
                    m = dict()
                    for point in self.sample[rv]:
                        m[point] = self.message_f_to_rv(point, f, rv, self.old_sample)
                    self.message_normalization(m)
                    self.message[(f, rv)] = m

            print(f'elapsed time {time.clock() - time_start}')

    def belief_rv(self, x, rv, sample):
        # sample is a set of sample points of neighbouring rvs
        # incoming message (from neighbouring rvs to neighbouring f) should be calculated before this process
        res = 1
        for f in rv.nb:
            res = res * self.message_f_to_rv(x, f, rv, sample)
        return res

    def belief(self, x, rv):
        if rv.value is None:
            b = self.belief_rv(x, rv, self.old_sample)
            z = quad(lambda val: self.belief_rv(val, rv, self.old_sample), -Inf, Inf)[0]
            return b / z
        else:
            return 1 if x == rv.value else 0

    def map(self, rv):
        if rv.value is None:
            res = fmin(lambda val: -self.belief_rv(val, rv, self.old_sample), 0, disp=False)[0]
            return res
        else:
            return rv.value
