from KalmanFilter import KalmanFilter
from Graph import *
from HybridLBPLogVersion import HybridLBP
from EPBPLogVersion import EPBP
from GaLBP import GaLBP
from GaBP import GaBP
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt


cluster_mat = scipy.io.loadmat('Data/cluster_NcutDiscrete.mat')['NcutDiscrete']
well_t = scipy.io.loadmat('Data/well_t.mat')['well_t']
ans = scipy.io.loadmat('Data/LRKF_tree.mat')['res']
param = scipy.io.loadmat('Data/LRKF_tree.mat')['param']
print(well_t.shape)

well_t = well_t[:, 199:]
well_t[well_t[:, 0] == 5000, 0] = 0
well_t[well_t == 5000] = 1
t = 20

cluster_id = [1]

rvs_id = []
for i in cluster_id:
    rvs_id.append(np.where(cluster_mat[:, i] == 1)[0])

rvs_id = np.concatenate(rvs_id, axis=None)
n_sum = len(rvs_id)
data = well_t[rvs_id, :t]

domain = Domain((-4, 4), continuous=True, integral_points=linspace(-4, 4, 30))

num_test = param.shape[1]

result = np.zeros([n_sum, num_test])
ans2 = np.zeros([n_sum, num_test])
time_cost = list()
for i in range(num_test):
    kmf = KalmanFilter(domain,
                       np.eye(n_sum) * param[2, i],
                       param[0, i],
                       np.eye(n_sum),
                       param[1, i])

    g, rvs_table = kmf.grounded_graph(t, data)
    bp = EPBP(g, n=50, proposal_approximation='simple')
    print('number of vr', len(g.rvs))
    num_evidence = 0
    for rv in g.rvs:
        if rv.value is not None:
            num_evidence += 1
    print('number of evidence', num_evidence)

    start_time = time.process_time()
    bp.run(20, log_enable=False)
    time_cost.append(time.process_time() - start_time)
    print('time lapse', time.process_time() - start_time)

    for idx, rv in enumerate(rvs_table[t - 1]):
        result[idx, i] = bp.map(rv)

    bp = GaLBP(g)
    bp.run(20, log_enable=False)

    for idx, rv in enumerate(rvs_table[t - 1]):
        ans2[idx, i] = bp.map(rv)

    print(f'avg err {np.average(result[:, i] - ans[:, i])}')
    print(f'avg err2 {np.average(result[:, i] - ans2[:, i])}')

err = abs(result - ans)
err = np.average(err, axis=0)

err2 = abs(result - ans)
err2 = np.average(err2, axis=0)

print('########################')
print(f'avg time {np.average(time_cost)}')
print(f'avg err {np.average(err)}')
print(f'err std {np.std(err)}')
print(f'avg err2 {np.average(err2)}')
print(f'err std2 {np.std(err2)}')
