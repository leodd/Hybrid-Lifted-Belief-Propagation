from Potential import GaussianPotential
from HybridLBP import *
from numpy import Inf


domain = Domain((-10, 10), continuous=True)

n = 10

rv = []
for _ in range(n):
    rv.append(RV(domain))

p1 = GaussianPotential([1, 2], [[2.0, 0.3], [0.3, 0.5]])

f = []
for i in range(n - 1):
    f.append(F(p1, (rv[i], rv[i+1])))

g = Graph()
g.rvs = rv
g.factors = f
g.init_nb()

bp = HybridLBP(g, 10)
bp.run(10)

for x in rv:
    # print(bp.belief(1, x))
    print(bp.map(x))
