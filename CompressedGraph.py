from Graph import *
from collections import Counter
from statistics import mean
from random import uniform
import numpy as np


class SuperRV:
    def __init__(self, rvs, domain=None, value=None):
        self.rvs = rvs
        self.domain = next(iter(rvs)).domain if domain is None else domain
        self.value = self.get_value(rvs) if value is None and not (next(iter(rvs)).value is None) else value
        self.nb = None
        self.count = None
        for rv in rvs:
            rv.cluster = self

    def __lt__(self, other):
        return hash(self) < hash(other)

    @staticmethod
    def get_value(rvs):
        temp = 0
        for rv in rvs:
            temp += rv.value
        return temp / len(rvs)

    @staticmethod
    def get_cluster(instance):
        return instance.cluster

    @staticmethod
    def get_rv_value(rv):
        return rv.value

    def get_variance(self):
        return np.var(tuple(map(self.get_rv_value, self.rvs)))

    def update_nb(self):
        rv = next(iter(self.rvs))
        self.count = Counter(map(self.get_cluster, rv.nb))
        self.nb = tuple(self.count)

    def split_by_structure(self):
        clusters = dict()
        for rv in self.rvs:
            signature = tuple(sorted(map(self.get_cluster, rv.nb)))
            if signature in clusters:
                clusters[signature].add(rv)
            else:
                clusters[signature] = {rv}

        res = set()
        i = iter(clusters)

        # reuse THIS instance
        self.rvs = clusters[next(i)]
        if self.value is not None:
            self.value = self.get_value(self.rvs)
        res.add(self)

        for _ in range(1, len(clusters)):
            new_rv = SuperRV(clusters[next(i)], self.domain, None)
            res.add(new_rv)

        for rv in res:
            rv.update_nb()

        return res

    def split_by_evidence(self, k=2, iteration=10):
        # can only be split into two when it is evidence rv, and there are multiple rvs
        if len(self.rvs) <= 1:
            return {self}

        # store all values
        values = Counter()
        for rv in self.rvs:
            values[rv.value] += 1

        if len(values) < k:
            k = len(values)
            if k <= 1:
                return {self}

        # initialize centroid
        centroids = []
        i = iter(values)
        for _ in range(k):
            centroids.append(next(i))
        centroids = np.array(centroids)

        # k-mean
        sum_table = np.zeros((k, 2))
        for _ in range(iteration):
            for v, n in values.items():
                idx = (np.abs(centroids - v)).argmin()
                sum_table[idx, 0] += v * n
                sum_table[idx, 1] += n
            for idx in range(k):
                centroids[idx] = sum_table[idx, 0] / sum_table[idx, 1]
            sum_table.fill(0)

        # clustering
        clusters = [set() for _ in range(k)]
        for rv in self.rvs:
            idx = (np.abs(centroids - rv.value)).argmin()
            clusters[idx].add(rv)

        res = set()
        # reuse THIS super rv instance
        self.rvs = clusters[0]
        self.value = centroids[0]
        res.add(self)

        for idx in range(1, k):
            res.add(SuperRV(clusters[idx], self.domain, centroids[idx]))

        for rv in res:
            rv.update_nb()

        return res


class SuperF:
    def __init__(self, factors):
        self.factors = factors
        self.potential = next(iter(factors)).potential
        self.nb = None
        for f in factors:
            f.cluster = self

    def __lt__(self, other):
        return hash(self) < hash(other)

    @staticmethod
    def get_cluster(instance):
        return instance.cluster

    def update_nb(self):
        f = next(iter(self.factors))
        self.nb = tuple(map(self.get_cluster, f.nb))

    def split_by_structure(self):
        clusters = dict()
        for f in self.factors:
            signature = tuple(sorted(map(self.get_cluster, f.nb))) \
                if f.potential.symmetric else tuple(map(self.get_cluster, f.nb))
            if signature in clusters:
                clusters[signature].add(f)
            else:
                clusters[signature] = {f}

        res = set()
        i = iter(clusters)

        # reuse THIS instance
        self.factors = clusters[next(i)]
        res.add(self)

        for _ in range(1, len(clusters)):
            res.add(SuperF(clusters[next(i)]))

        for f in res:
            f.update_nb()

        return res


class CompressedGraph:
    # color passing algorithm for compressing graph

    def __init__(self, graph):
        self.g = graph
        self.rvs = set()
        self.factors = set()
        self.continuous_evidence = set()

    def init_cluster(self, is_split_cont_evidence=True):
        self.rvs.clear()
        self.factors.clear()
        self.continuous_evidence.clear()

        # group rvs according to domain
        color_table = dict()
        for rv in self.g.rvs:
            if rv.domain in color_table:
                color_table[rv.domain].add(rv)
            else:
                color_table[rv.domain] = {rv}
        for domain, cluster in color_table.items():
            # split rvs according to if it is hidden
            hidden = set()
            for rv in cluster:
                if rv.value is None:
                    hidden.add(rv)
            if len(hidden) != 0:
                self.rvs.add(SuperRV(hidden))
            evidence = cluster - hidden
            if len(evidence) != 0:
                if not is_split_cont_evidence and domain.continuous:
                    # for continuous evidences, we simply cluster them together
                    # without considering the actual value
                    rv = SuperRV(evidence)
                    self.rvs.add(rv)
                    self.continuous_evidence.add(rv)
                else:
                    # for each evidence, we cluster them by their value
                    value_table = dict()
                    for e in evidence:
                        if e.value in value_table:
                            value_table[e.value].add(e)
                        else:
                            value_table[e.value] = {e}
                    for _, evidence_cluster in value_table.items():
                        self.rvs.add(SuperRV(evidence_cluster))

        # group factors according to potential
        color_table.clear()
        for f in self.g.factors:
            if f.potential in color_table:
                color_table[f.potential].add(f)
            else:
                color_table[f.potential] = {f}
        for _, cluster in color_table.items():
            self.factors.add(SuperF(cluster))

    def split_evidence(self, k=2, iteration=10):
        temp = set()
        for rv in self.continuous_evidence:
            new_rvs = rv.split_by_evidence(k, iteration)
            if len(new_rvs) > 1:
                temp |= new_rvs
        self.continuous_evidence = temp
        self.rvs |= temp

    def split_rvs(self):
        temp = set()
        temp_cont = set()
        for rv in self.rvs:
            new_rvs = rv.split_by_structure()
            temp |= new_rvs
            if rv.value is not None and rv in self.continuous_evidence:
                temp_cont |= new_rvs
        self.rvs = temp
        self.continuous_evidence = temp_cont

    def split_factors(self):
        temp = set()
        for f in self.factors:
            temp |= f.split_by_structure()
        self.factors = temp
