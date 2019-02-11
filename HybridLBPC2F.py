from CompressedGraph import *
from numpy import Inf, exp
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import fmin
from statistics import mean
from math import sqrt, log, e
from collections import Counter
from itertools import product
import random

import time


class HybridLBP:
    # Hybrid lifted particle belief propagation

    var_threshold = 5
    max_log_value = 700

    def __init__(self,
                 g,
                 n=50,
                 k_mean_k=2,
                 k_mean_iteration=10):
        self.g = CompressedGraph(g)
        self.n = n
        self.message = dict()  # log message, message in log space
        self.sample = dict()
        self.old_sample = dict()
        self.q = dict()
        self.eta_message = dict()  # each site distribution corresponds to a message (factor to variable)
        self.query_cache = dict()
        self.k_mean_k = k_mean_k
        self.k_mean_iteration = k_mean_iteration

    ###########################
    # utils
    ###########################

    @staticmethod
    def gaussian_product(*gaussian):
        # input a list of gaussian's mean and variance
        # output the product distribution's mean and variance
        mu, sig = 0, 0
        for g in gaussian:
            mu_, sig_, count = g
            sig += sig_ ** -1 * count
            mu += sig_ ** -1 * mu_ * count
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

    ###########################
    # EPBP functions
    ###########################

    def generate_sample(self):
        sample = dict()
        for rv in self.g.rvs:
            if rv.value is None:
                if rv.domain.continuous:
                    sample[rv] = norm(self.q[rv][0], sqrt(self.q[rv][1])).rvs(self.n)
                else:
                    sample[rv] = rv.domain.values
        return sample

    def initial_proposal(self):
        for rv in self.g.rvs:
            if rv.value is None and rv.domain.continuous:
                self.q[rv] = (0, 5)

                count = sum(rv.count.values())  # count the number of incoming messages
                site = (0, 5 * count)

                for f in rv.nb:
                    self.eta_message[(f, rv)] = site

    def update_proposal(self):
        for rv in self.g.rvs:
            if rv.value is None and rv.domain.continuous:
                eta = list()
                min_sig = sum(rv.count.values()) * self.var_threshold
                for f in rv.nb:
                    mu, sig = self.eta_approximation(f, rv)
                    if 0 < sig < Inf:
                        sig = max(sig, min_sig)
                        self.eta_message[(f, rv)] = (mu, sig)
                    else:
                        print('!')
                        mu, sig = self.eta_message[(f, rv)]
                    eta.append((mu, sig, rv.count[f]))
                # old_q = self.q[rv]
                self.q[rv] = self.gaussian_product(*eta)
                # print(f'{old_q} -> {self.q[rv]}')

    def eta_approximation(self, f, rv):
        # compute the cavity distribution
        cavity = self.gaussian_division(self.q[rv], self.eta_message[(f, rv)])

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

        # approximate eta
        return self.gaussian_division((mu, sig), cavity)

    def important_weight(self, x, rv):
        if rv.value is None and rv.domain.continuous:
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
                    res += self.message[(nb, rv)][x] * rv.count[nb]
            return res + log(self.important_weight(x, rv)) + self.message[(f, rv)][x] * (rv.count[f] - 1)

    def message_f_to_rv(self, x, f, rv, sample):
        # sample is a set of sample points of neighbouring rvs
        # incoming message should be calculated before this process
        res = 0
        param = []
        flag = True
        for nb in f.nb:
            if nb == rv and flag:
                param.append((x,))
                flag = False
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

    # def message_f_to_rv(self, x, f, rv, sample):
    #     # sample is a set of sample points of neighbouring rvs
    #     # incoming message should be calculated before this process
    #     all_m = []
    #     sum_of_count = 0
    #     for f_nb, count in f.fs.items():
    #         res = 0
    #         param = []
    #         flag = True
    #         for nb in f_nb:
    #             if nb == rv and flag:
    #                 param.append((x,))
    #                 flag = False
    #             elif nb.value is None:
    #                 param.append(sample[nb])
    #             else:
    #                 param.append((nb.value,))
    #
    #         if flag:
    #             continue
    #
    #         for x_join in product(*param):
    #             m = 0
    #             for idx, nb in enumerate(f_nb):
    #                 if nb != rv and nb.value is None:
    #                     m += self.message[(nb, f)][x_join[idx]]
    #             res += f.potential.get(x_join) * e ** m
    #
    #         all_m.append(((log(res) if res > 0 else -700), count))
    #         sum_of_count += count
    #
    #     res = 0
    #     for m, count in all_m:
    #         res += m * count
    #
    #     return res / sum_of_count

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

    @staticmethod
    def message_normalization(message):
        z = 0
        for k, v in message.items():
            z = z + v
        for k, v in message.items():
            message[k] = v / z

    ###########################
    # lifted graph functions
    ###########################

    def update_factor_nb(self, super_f):
        fs = Counter()
        for f in super_f.factors:
            nb = tuple(map(self.get_cluster, f.nb))
            fs[nb] += 1
        super_f.fs = fs

    ###########################
    # query functions
    ###########################

    @staticmethod
    def get_cluster(instance):
        return instance.cluster

    def belief_rv_query(self, x, rv, sample):
        res = 0
        for f in rv.nb:
            res += self.message_f_to_rv(x, f.cluster, rv.cluster, sample)
        return res

    def belief(self, x, rv):
        if rv.value is None:
            signature = tuple(sorted(map(self.get_cluster, rv.nb)))

            if rv.domain.continuous:
                if signature in self.query_cache:
                    z = self.query_cache[signature]
                else:
                    z = quad(lambda val: e ** self.belief_rv_query(val, rv, self.sample), -Inf, Inf)[0]
                    self.query_cache[signature] = z

                b = e ** self.belief_rv_query(x, rv, self.sample)

                return b / z
            else:
                if signature in self.query_cache:
                    b = self.query_cache[signature]
                else:
                    b = dict()
                    for v in rv.domain.values:
                        b[v] = e ** self.belief_rv_query(v, rv, self.sample)
                    self.message_normalization(b)
                    self.query_cache[signature] = b

                return b[x]
        else:
            return 1 if x == rv.value else 0

    def map(self, rv):
        if rv.value is None:
            signature = tuple(sorted(map(self.get_cluster, rv.nb)))

            if rv.domain.continuous:
                if signature in self.query_cache:
                    res = self.query_cache[signature]
                else:
                    res = fmin(lambda val: -self.belief_rv_query(val, rv, self.sample), 0, disp=False)[0]
                    self.query_cache[signature] = res

                # print(f'{self.q[rv.cluster]}')
                return res
            else:
                if signature in self.query_cache:
                    max_x = self.query_cache[signature]
                else:
                    max_b = -Inf
                    max_x = None
                    for x in rv.domain.values:
                        b = self.belief_rv_query(x, rv, self.sample)
                        (max_b, max_x) = (b, x) if b > max_b else (max_b, max_x)
                    self.query_cache[signature] = max_x

                return max_x
        else:
            return rv.value

    ###########################
    # main running function
    ###########################

    def run(self, iteration=10, log_enable=False):
        # initialize cluster
        self.g.init_cluster(False)

        for _ in range(10):
            self.g.split_factors()
            self.g.split_rvs()

        while len(self.g.continuous_evidence) > 0:
            self.g.split_evidence(3, 50)
            self.g.split_factors()
            self.g.split_rvs()

        # for _ in range(20):
        #     self.g.split_factors()
        #     self.g.split_rvs()

        for f in self.g.factors:
            self.update_factor_nb(f)

        # initialize proposal
        self.initial_proposal()

        # poll sample from the initial distribution
        self.sample = self.generate_sample()

        # initialize log message to 0
        for rv in self.g.rvs:
            if rv.value is None:
                for f in rv.nb:
                    m = {k: 0 for k in self.sample[rv]}
                    eta_m = {k: 0 for k in rv.domain.integral_points}
                    self.message[(f, rv)] = {**m, **eta_m}
                    self.message[(rv, f)] = m

        # LBP iteration
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
                # poll new sample
                self.old_sample = self.sample
                self.sample = self.generate_sample()

                # calculate messages from f to rv
                for f in self.g.factors:
                    for rv in f.nb:
                        if rv.value is None:
                            # compute the message for each sample point
                            m = dict()
                            for point in self.sample[rv]:
                                m[point] = self.message_f_to_rv(point, f, rv, self.old_sample)
                            for point in rv.domain.integral_points:
                                m[point] = self.message_f_to_rv(point, f, rv, self.old_sample)
                            self.message[(f, rv)] = m

                if log_enable:
                    print(f'\tf to rv {time.clock() - time_start}')
                    time_start = time.clock()

                # update proposal
                self.update_proposal()
                if log_enable:
                    print(f'\tproposal {time.clock() - time_start}')
