import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.io


data = np.load('Data/well_t_prediction_2.npy')
t = data.shape[0]
for idx in range(data.shape[1]):
    y = []
    for i in range(t):
        y.append(data[i, idx, 1])

    plt.plot(list(range(t)), y)

data = scipy.io.loadmat('Data/LRKF_well_t_prediction_2.mat')['x']
for idx in range(data.shape[0]):
    y = []
    for i in range(t):
        y.append(data[idx, i])

    plt.plot(list(range(t)), y, color='black', linestyle='--', dashes=(2, 5))

custom_lines = [Line2D([0], [0], color='black', lw=2),
                Line2D([0], [0], color='black', lw=2, linestyle='--', dashes=(2, 2))]
plt.legend(custom_lines, ('LHBP', 'LRKF (exact)'), loc='lower left')

plt.xlabel('time step')
plt.ylabel('flow head position')

plt.xticks((0,4,9,14,19), (1,5,10,15,20))

plt.show()

# cluster_mat = scipy.io.loadmat('Data/cluster_NcutDiscrete.mat')['NcutDiscrete']
# well_t = scipy.io.loadmat('Data/well_t.mat')['well_t']
#
# well_t = well_t[:, 199:]
#
# cluster_id = [1]
#
# cluster = []
# for i in cluster_id:
#     cluster.append(np.where(cluster_mat[:, i] == 1)[0])
#
# for c_id, c in zip(cluster_id, cluster):
#     for well_id in c:
#         y = []
#         x = []
#         for i in range(t):
#             if well_t[well_id, i] != 5000:
#                 y.append(well_t[well_id, i])
#                 x.append(i)
#         plt.plot(x, y)
# plt.show()
