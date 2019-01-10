from Graph import Potential
import numpy as np
from math import pow, pi, e, sqrt
from scipy.stats import norm


class TablePotential(Potential):
    def __init__(self, table, symmetric=False):
        Potential.__init__(self, symmetric=symmetric)
        self.table = table

    def get(self, parameters):
        return self.table[parameters]


class GaussianPotential(Potential):
    def __init__(self, mu, sig, w=1):
        Potential.__init__(self, symmetric=False)
        self.mu = np.array(mu)
        self.sig = np.matrix(sig)
        self.inv = self.sig.I
        det = np.linalg.det(self.sig)
        p = float(len(mu))
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        self.coefficient = w / (pow(2*pi, p*0.5) * pow(det, 0.5))

    def get(self, parameters):
        x_mu = np.matrix(np.array(parameters) - self.mu)
        return self.coefficient * pow(e, -0.5 * (x_mu * self.inv * x_mu.T))


class ImageNodePotential(Potential):
    def __init__(self, mu, sig):
        Potential.__init__(self, symmetric=True)
        self.mu = mu
        self.sig = sig

    def get(self, parameters):
        return norm(self.mu, self.sig).pdf(parameters[0] - parameters[1])


# class ImageEdgePotential:
#     def __init__(self, mu, sig):
#         self.mu = mu
#         self.sig = sig
#
#     def get(self, parameters):
#         return 0.1 + 20 * norm(self.mu, self.sig).pdf(parameters[0] - parameters[1])


class ImageEdgePotential(Potential):
    def __init__(self, distant_cof, scaling_cof, max_threshold):
        Potential.__init__(self, symmetric=True)
        self.distant_cof = distant_cof
        self.scaling_cof = scaling_cof
        self.max_threshold = max_threshold
        self.v = pow(e, -self.max_threshold / self.scaling_cof)

    def get(self, parameters):
        d = abs(parameters[0] - parameters[1])
        if d > self.max_threshold:
            return d * self.distant_cof + self.v
        else:
            return d * self.distant_cof + pow(e, -d / self.scaling_cof)
