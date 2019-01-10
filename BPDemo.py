from Graph import *
from BP import *
from Potential import TablePotential


domain = Domain((True, False))

a = RV(domain, False)
b = RV(domain)
c = RV(domain, False)

p1 = TablePotential({
    (True, True): 4,
    (True, False): 1,
    (False, True): 1,
    (False, False): 3
})

fab = F(p1, (a, b))
fbc = F(p1, (b, c))

g = Graph()
g.rvs = [a, b, c]
g.factors = [fab, fbc]
g.init_nb()

bp = BP(g)
bp.run(10)

print(bp.belief(a))
print(bp.belief(b))
print(bp.belief(c))
