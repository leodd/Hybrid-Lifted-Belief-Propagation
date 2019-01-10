import pandas as pd
from Potential import *
from PBP import PBP
from numpy import Inf
import numpy as np
from Graph import *
from show_image import show_images

row = 50
col = 50

data = pd.read_fwf('./noisyImage.dat', header=None)
m = data.iloc[0:row, 0:col].values
m = m * 100

# show_images((m,), vmin=-30, vmax=130)

domain = Domain((-30, 130), continuous=True)

rvs = []
for _ in range(row * col):
    rvs.append(RV(domain))

fs = []

# create node factors
for i in range(row):
    for j in range(col):
        fs.append(
            F(
                ImageNodePotential(m[i, j], 0, 5),
                (rvs[i * col + j],)
            )
        )

# create edge factors
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
g.rvs = rvs
g.factors = fs
g.init_nb()

pbp = PBP(g, 10)


def initial_proposal():
    for i in range(row):
        for j in range(col):
            pbp.q[rvs[i * col + j]] = (m[i, j], 2)


pbp.custom_initial_proposal = initial_proposal
pbp.run(20)

# reconstruct image
m_hat = np.zeros((row, col))
for i in range(row):
    for j in range(col):
        m_hat[i, j] = pbp.map(rvs[i * col + j])

show_images((m, m_hat), vmin=-30, vmax=130)
