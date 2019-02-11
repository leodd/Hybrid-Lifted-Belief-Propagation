from KalmanFilter import KalmanFilter
from Graph import *
from HybridLBPLogVersion import HybridLBP
from EPBPLogVersion import EPBP
from GaBP import GaBP
import numpy as np
import scipy.io
import time


cluster_mat = scipy.io.loadmat('Data/cluster_NcutDiscrete.mat')['NcutDiscrete']
well_t = scipy.io.loadmat('Data/well_t.mat')['well_t']

well_t = well_t[:, 200:]
t = 20

cluster_id = [0, 1]

rvs_id = []
for i in cluster_id:
    rvs_id.append(np.where(cluster_mat[:, i] == 1)[0])

rvs_id = np.concatenate(rvs_id, axis=None)
data = well_t[rvs_id, :]

domain = Domain((-100, 150), continuous=True, integral_points=linspace(-100, 150, 50))

kmf = KalmanFilter(domain, np.eye(len(rvs_id)), np.eye(len(rvs_id)), np.eye(len(rvs_id)), np.eye(len(rvs_id)))
g, rvs_table = kmf.grounded_graph(t, data)

print('number of vr', len(g.rvs))
num_evidence = 0
for rv in g.rvs:
    if rv.value is not None:
        num_evidence += 1
print('number of evidence', num_evidence)

start_time = time.time()
bp = HybridLBP(g, n=100)
bp.run(20, log_enable=False)
print('time lapse', time.time() - start_time)

result = []
for i in range(t):
    temp = []
    for idx, rv in enumerate(rvs_table[i]):
        # print(key, bp.map(rv))
        temp.append([idx, bp.map(rv)])
    result.append(temp)

np.save('Data/well_t_prediction', np.array(result))
