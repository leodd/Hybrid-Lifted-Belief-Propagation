from RelationalGraph import *
from MLNPotential import *
from HybridLBPLogVersion import HybridLBP
from EPBPLogVersion import EPBP
import numpy as np
import time


X = np.load('Data/smoker_instances.npy')
D = np.load('Data/smoker_data.npy')

people = []
for p in X:
    people.append(p[1])

data = dict()
for line in D:
    key = tuple([x.strip() for x in line[0].split(',')])
    data[key] = float(line[1])

domain_bool = Domain((0, 1))
domain_real = Domain((0, 1), continuous=True, integral_points=linspace(0, 1, 10))

lv_x = LV(people)
lv_y = LV(people)

atom_smoke = Atom(domain_real, logical_variables=(lv_x,), name='smoke')
atom_smoke_y = Atom(domain_real, logical_variables=(lv_y,), name='smoke')
atom_cancer = Atom(domain_real, logical_variables=(lv_x,), name='cancer')
atom_friend = Atom(domain_bool, logical_variables=(lv_x, lv_y), name='friend')

f1 = ParamF(MLNPotential(lambda x: imp_op(x[0], x[1]), w=1), nb=(atom_smoke, atom_cancer))
f2 = ParamF(
    MLNPotential(lambda x: imp_op(x[0], bic_op(x[1], x[2])), w=1),
    nb=(atom_friend, atom_smoke, atom_smoke_y),
    constrain=lambda s: s[lv_x] < s[lv_y]
)
f3 = ParamF(MLNPotential(lambda x: imp_op(1 - x[0], 1 - x[1]), w=1), nb=(atom_smoke, atom_cancer))
f4 = ParamF(MLNNodePotential(), nb=(atom_smoke,))
f5 = ParamF(MLNNodePotential(), nb=(atom_cancer,))

rel_g = RelationalGraph()
rel_g.atoms = (atom_smoke, atom_cancer, atom_friend)
rel_g.param_factors = (f1, f2)
rel_g.data = data

rel_g.init_nb()
g, rvs_table = rel_g.grounded_graph()
print('number of vr', len(g.rvs))
num_evidence = 0
for rv in g.rvs:
    if rv.value is not None:
        num_evidence += 1
print('number of evidence', num_evidence)

bp = EPBP(g, n=20, proposal_approximation='simple')
start_time = time.process_time()
bp.run(10, log_enable=False)
print('time', time.process_time() - start_time)

for key, rv in rvs_table.items():
    v = bp.map(rv)
    p = list()
    for i in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        p.append(bp.belief(i, rv))
    print(key, v, p)
