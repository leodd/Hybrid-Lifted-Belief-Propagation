import numpy as np
import matplotlib.pyplot as plt
import scipy.io


data = np.load('Data/well_t_prediction.npy')
t = data.shape[0]
for idx in range(100):
    y = []
    for i in range(t):
        y.append(data[i, idx, 1])

    plt.plot(list(range(t)), y)
plt.show()

cluster_mat = scipy.io.loadmat('Data/cluster_NcutDiscrete.mat')['NcutDiscrete']
well_t = scipy.io.loadmat('Data/well_t.mat')['well_t']

well_t = well_t[:, 200:]

cluster_id = [0, 1]

cluster = []
for i in cluster_id:
    cluster.append(np.where(cluster_mat[:, i] == 1)[0])

for c_id, c in zip(cluster_id, cluster):
    for well_id in c:
        y = []
        x = []
        for i in range(t):
            if well_t[well_id, i] != 5000:
                y.append(well_t[well_id, i])
                x.append(i)
        plt.plot(x, y)
plt.show()
