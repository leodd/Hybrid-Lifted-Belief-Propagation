from RelationalDynamicModel import *
from HybridLBPLogVersion import HybridLBP
from EPBPLogVersion import EPBP
from GaBP import GaBP
import numpy as np
import scipy.io
import time


cluster_mat = scipy.io.loadmat('Data/cluster_NcutDiscrete.mat')['NcutDiscrete']
well_t = scipy.io.loadmat('Data/well_t.mat')['well_t']

well_t = well_t[:, 200:]

cluster_id = [0, 1]
t = 100

cluster = []
for i in cluster_id:
    cluster.append(np.where(cluster_mat[:, i] == 1)[0])

data = dict()
for c_id, c in zip(cluster_id, cluster):
    for well_id in c:
        for i in range(t):
            if well_t[well_id, i] != 5000:
                data[(c_id, well_id, i)] = well_t[well_id, i]

print(data)

domain = Domain((-100, 150), continuous=True, integral_points=linspace(-100, 150, 50))

atoms = []
for c_id, c in zip(cluster_id, cluster):
    atoms.append(
        Atom(domain,
             (LV(c),),
             name=c_id)
    )


class GaussianPotential(Potential):
    def __init__(self, coeff, sig):
        Potential.__init__(self, symmetric=False)
        self.coeff = coeff
        self.sig = sig * 2

    def get(self, parameters):
        return np.exp(-(parameters[1] - self.coeff * parameters[0]) ** 2 / self.sig )


pairwise_p = GaussianPotential(1, 50)
observe_p = GaussianPotential(1, 25)
indirect_observe_p = GaussianPotential(1, 500)
transition_p = GaussianPotential(1, 20)

pairwise_factor = ParamF(pairwise_p, [atoms[0], atoms[1]])

rel_dynamic_g = RelationalDynamicModel()
rel_dynamic_g.atoms = atoms
rel_dynamic_g.pairwise_factors = (pairwise_factor,)
rel_dynamic_g.observe_potentials = {0: observe_p, 1: observe_p}
rel_dynamic_g.indirect_observe_potentials = {0: indirect_observe_p, 1: indirect_observe_p}
rel_dynamic_g.transition_potentials = {0: transition_p, 1: transition_p}
rel_dynamic_g.data = data

rel_dynamic_g.init_nb()
g, rvs_table = rel_dynamic_g.grounded_graph(t)

print('number of vr', len(g.rvs))
num_evidence = 0
for rv in g.rvs:
    if rv.value is not None:
        num_evidence += 1
print('number of evidence', num_evidence)

bp = HybridLBP(g, n=20)
bp.run(20, log_enable=False)

for key, rv in rvs_table[t-1].items():
    print(key, bp.map(rv))
