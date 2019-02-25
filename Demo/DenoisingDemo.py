import pandas as pd
from Potential import *
from Graph import *
from HybridLBPLogVersion import HybridLBP
from EPBPLogVersion import EPBP
import numpy as np
from show_image import show_images

import time


row = 50
col = 50

data = pd.read_fwf('../Data/noisyImage.dat', header=None)
m = data.iloc[0:row, 0:col].values
m = m * 100

# show_images((m,), vmin=-30, vmax=130)

domain = Domain((-30, 130), continuous=True)

evidence = [None] * (col * row)
for i in range(row):
    for j in range(col):
        evidence[i * col + j] = RV(domain, m[i, j])

rvs = []
for _ in range(row * col):
    rvs.append(RV(domain))

fs = []

# create hidden-obs factors
pxo = ImageNodePotential(0, 5)
for i in range(row):
    for j in range(col):
        fs.append(
            F(
                pxo,
                (rvs[i * col + j], evidence[i * col + j])
            )
        )

# create hidden-hidden factors
pxy = ImageEdgePotential(0, 3.5, 25)
for i in range(row):
    for j in range(col - 1):
        fs.append(
            F(
                pxy,
                (rvs[i * col + j], rvs[i * col + j + 1])
            )
        )
for i in range(row - 1):
    for j in range(col):
        fs.append(
            F(
                pxy,
                (rvs[i * col + j], rvs[(i + 1) * col + j])
            )
        )

g = Graph()
g.rvs = rvs + evidence
g.factors = fs
g.init_nb()

bp = HybridLBP(g, n=10, proposal_approximation='simple')

# def initial_proposal():
#     for i in range(row):
#         for j in range(col):
#             bp.q[rvs[i * col + j]] = (m[i, j], 2)
#
# bp.custom_initial_proposal = initial_proposal

start_time = time.process_time()
bp.run(10, c2f=0, log_enable=False)
print('time', time.process_time() - start_time)

print(len(bp.g.rvs))

# reconstruct image
m_hat = np.zeros((row, col))
for i in range(row):
    for j in range(col):
        m_hat[i, j] = bp.map(rvs[i * col + j])

show_images((m, m_hat), vmin=-30, vmax=130)
