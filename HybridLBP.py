from CompressedGraph import *
from numpy import Inf
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import fmin
from math import sqrt
from itertools import product

import time


class HybridLBP:
    # Hybrid lifted particle belief propagation

    var_threshold = 0.2

    def __init__(self,
                 g,
                 n=50,
                 step_size=0.2,
                 k_mean_k=2,
                 k_mean_iteration=3):
        self.g = CompressedGraph(g)
        self.n = n
        self.step_size = step_size
        self.message = dict()
        self.sample = dict()
        self.q = dict()
        self.query_cache = dict()
        self.custom_initial_proposal = self.initial_proposal
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
            else:
                sample[rv] = (rv.value,)
        return sample

    def initial_proposal(self):
        for rv in self.g.rvs:
            if rv.value is None and rv.domain.continuous:
                average = sum(rv.domain.values) / 2
                self.q[rv] = (average, 10)
            else:
                self.q[rv] = None

    def update_proposal(self):
        for rv in self.g.rvs:
            if rv.value is None and rv.domain.continuous:
                # time_start = time.clock()
                eta = list()
                for f in rv.nb:
                    mu, sig = self.eta_message_f_to_rv(f, rv)
                    eta.append((mu, sig, rv.count[f]))
                mu, sig = self.gaussian_product(*eta)
                old_mu, old_sig = self.q[rv]
                mu = old_mu + self.step_size * (mu - old_mu)
                sig = old_sig + self.step_size * (sig - old_sig)
                if sig < self.var_threshold:
                    sig = self.var_threshold
                self.q[rv] = (mu, sig)
            else:
                self.q[rv] = None

    def important_weight(self, x, rv):
        if rv.value is None and rv.domain.continuous:
            return 1 / (self.n * norm(self.q[rv][0], sqrt(self.q[rv][1])).pdf(x))
        else:
            return 1

    def eta_message_f_to_rv(self, f, rv):
        # eta_message is the gaussian approximation of the particle message
        # the return is a tuple of mean and variance
        points_val = []
        z = 0
        mu = 0
        var = 0

        for x in rv.domain.integral_points:
            points_val.append(self.message[(f, rv)][x])
            z = z + points_val[-1]

        for i in range(len(rv.domain.integral_points)):
            mu = mu + (points_val[i] * rv.domain.integral_points[i]) / z
            var = var + (points_val[i] * rv.domain.integral_points[i] ** 2) / z

        var = var - mu ** 2

        return mu, (var if var > self.var_threshold else self.var_threshold)

    def message_rv_to_f(self, x, rv, f):
        # the incoming message on x must be calculated before calculating the out going message
        # that means x should not be re-sampled
        # (before calculating out going message and after calculating incoming message)
        if rv.value is None:
            res = 1
            for nb in rv.nb:
                if nb != f:
                    res = res * (self.message[(nb, rv)][x] ** rv.count[nb])
            return res * self.important_weight(x, rv) * (self.message[(f, rv)][x] ** (rv.count[f] - 1))
        else:
            return 1

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
            else:
                param.append(sample[nb])

        for x_join in product(*param):
            m = 1
            for i in range(len(f.nb)):
                if f.nb[i] != rv:
                    m = m * self.message[(f.nb[i], f)][x_join[i]]
            res = res + f.potential.get(x_join) * m

        return res

    def belief_rv(self, x, rv, sample):
        # sample is a set of sample points of neighbouring rvs
        # incoming message (from neighbouring rvs to neighbouring f) should be calculated before this process
        res = 1
        for f in rv.nb:
            res *= self.message_f_to_rv(x, f, rv, sample) ** rv.count[f]
        return res

    @staticmethod
    def message_normalization(message):
        z = 0
        for k, v in message.items():
            z = z + v
        for k, v in message.items():
            message[k] = v / z

    ###########################
    # color passing functions
    ###########################

    def split_evidence(self, k=2, iteration=3):
        temp = set()
        for rv in self.g.continuous_evidence:
            # split evidence
            new_rvs = rv.split_by_evidence(k, iteration)

            if len(new_rvs) > 1:
                for new_rv in new_rvs:
                    # update sample
                    self.sample[new_rv] = (new_rv.value,)
                temp |= new_rvs

        self.g.continuous_evidence = temp
        self.g.rvs |= temp

    def split_rvs(self):
        temp = set()
        for rv in self.g.rvs:
            # split rvs
            new_rvs = rv.split_by_structure()

            if len(new_rvs) > 1:
                for new_rv in new_rvs:
                    # update message
                    if new_rv.value is None:
                        for f in new_rv.nb:
                            self.message[(f, new_rv)] = self.message[(f, rv)]
                        # update proposal
                        self.q[new_rv] = self.q[rv]

                    # update sample
                    self.sample[new_rv] = self.sample[rv]
            temp |= new_rvs
        self.g.rvs = temp

    def split_factors(self):
        temp = set()
        for f in self.g.factors:
            # split factors
            new_fs = f.split_by_structure()

            for new_f in new_fs:
                # update message
                for rv in new_f.nb:
                    self.message[(rv, new_f)] = self.message[(rv, f)]
            temp |= new_fs
        self.g.factors = temp

    ###########################
    # query functions
    ###########################

    @staticmethod
    def get_cluster(instance):
        return instance.cluster

    def belief_rv_query(self, x, rv, sample):
        res = 1
        for f in rv.nb:
            res = res * self.message_f_to_rv(x, f.cluster, rv.cluster, sample)
        return res

    def belief(self, x, rv):
        if rv.value is None:
            signature = tuple(sorted(map(self.get_cluster, rv.nb)))

            if rv.domain.continuous:
                if signature in self.query_cache:
                    z = self.query_cache[signature]
                else:
                    z = quad(lambda val: self.belief_rv_query(val, rv, self.sample), -Inf, Inf)[0]
                    self.query_cache[signature] = z

                b = self.belief_rv_query(x, rv, self.sample)

                return b / z
            else:
                if signature in self.query_cache:
                    b = self.query_cache[signature]
                else:
                    b = dict()
                    for v in rv.domain.values:
                        b[v] = self.belief_rv_query(v, rv, self.sample)
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

    def run(self, iteration=10, k_mean_k=2, k_mean_iteration=3, log_enable=False):
        # initialize cluster
        self.g.init_cluster()
        self.g.split_evidence(self.k_mean_k, self.k_mean_iteration)
        self.g.split_factors()
        self.g.split_rvs()

        # initialize proposal
        self.custom_initial_proposal()

        # poll sample from the initial distribution
        self.sample = self.generate_sample()

        # initialize message to 1
        for rv in self.g.rvs:
            for f in rv.nb:
                m = {k: 1 for k in self.sample[rv]}
                eta_m = {k: 1 for k in rv.domain.integral_points}
                self.message[(f, rv)] = {**m, **eta_m}
                self.message[(rv, f)] = m

        # LBP iteration
        for i in range(iteration):
            print(f'iteration: {i + 1}')
            if log_enable:
                time_start = time.clock()

            if i > 0:
                self.split_evidence(self.k_mean_k, self.k_mean_iteration)
                if log_enable:
                    print(f'\tevidence {time.clock() - time_start}')
                    time_start = time.clock()
                self.split_rvs()
                if log_enable:
                    print(f'\tsplit rv {time.clock() - time_start}')
                    time_start = time.clock()

            # calculate messages from rv to f
            for rv in self.g.rvs:
                for f in rv.nb:
                    # compute the message for each sample point
                    m = dict()
                    for point in self.sample[rv]:
                        m[point] = self.message_rv_to_f(point, rv, f)
                    self.message_normalization(m)
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

                self.split_factors()
                if log_enable:
                    print(f'\tsplit factor {time.clock() - time_start}')
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
                            # self.message_normalization(m)
                            # compute the eta message for each integral point
                            for point in rv.domain.integral_points:
                                m[point] = self.message_f_to_rv(point, f, rv, old_sample)
                            self.message[(f, rv)] = m

                if log_enable:
                    print(f'\tf to rv {time.clock() - time_start}')

        self.split_factors()
