from KalmanFilter import KalmanFilter
from Graph import *
from HybridLBPLogVersion import HybridLBP
from EPBPLogVersion import EPBP
from GaBP import GaBP
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt


cluster_mat = scipy.io.loadmat('Data/cluster_NcutDiscrete.mat')['NcutDiscrete']
well_t = scipy.io.loadmat('Data/well_t.mat')['well_t']

well_t = well_t[:, 199:]
well_t[well_t == 5000] = 0
t = 6

cluster_id = [1]

rvs_id = []
for i in cluster_id:
    rvs_id.append(np.where(cluster_mat[:, i] == 1)[0])

rvs_id = np.concatenate(rvs_id, axis=None)
# rvs_id = rvs_id[:6]
data = well_t[rvs_id, :t]

# for i in range(len(rvs_id)):
#     plt.plot(list(range(20)), data[i, :])
# plt.show()

domain = Domain((-10, 10), continuous=True, integral_points=linspace(-10, 10, 30))

kmf = KalmanFilter(domain,
                   # np.ones([len(rvs_id), len(rvs_id)]),
                   np.eye(len(rvs_id)) + 0.1,
                   100,
                   np.eye(len(rvs_id)),
                   1)

result = []
for i in range(t):
    # i = t - 1
    g, rvs_table = kmf.grounded_graph(i + 1, data)
    bp = HybridLBP(g, n=20, proposal_approximation='simple')
    print('number of vr', len(g.rvs))
    num_evidence = 0
    for rv in g.rvs:
        if rv.value is not None:
            num_evidence += 1
    print('number of evidence', num_evidence)

    start_time = time.time()
    bp.run(6, log_enable=False)
    print('time lapse', time.time() - start_time)

    # for i in range(t):
    temp = []
    for idx, rv in enumerate(rvs_table[i]):
        temp.append([idx, bp.map(rv)])
    result.append(temp)

np.save('Data/well_t_prediction', np.array(result))
