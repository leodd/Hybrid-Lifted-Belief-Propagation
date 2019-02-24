from RelationalGraph import *
from MLNPotential import *
from HybridLBPLogVersion import HybridLBP
from EPBPLogVersion import EPBP
import numpy as np
import time


num_people = 100

people = []
for p in range(num_people):
    people.append(f'p{p}')

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

rel_g = RelationalGraph()
rel_g.atoms = (atom_smoke, atom_cancer, atom_friend)
rel_g.param_factors = (f1, f2)
rel_g.init_nb()

data = dict()

for a in range(num_people):
    idx = np.random.choice(num_people, int(num_people * 0.1), replace=False)
    for b in idx:
        if a < b:
            data[('friend', f'p{a}', f'p{b}')] = np.random.choice([0, 1])

A = np.random.choice(num_people, int(num_people * 0.1), replace=False)
for a in A:
    data[('smoke', f'p{a}')] = np.random.random_sample()

rel_g.data = data
g, rvs_table = rel_g.grounded_graph()

print('number of vr', len(g.rvs))
print('number of evidence', len(data))

key_list = list()
for p in range(num_people):
    key_list.append(('cancer', f'p{p}'))

num_test = 5

avg_diff = dict()
variance = dict()
time_cost = dict()

ans = dict()

name = 'EPBP'
res = np.zeros((len(key_list), num_test))
for j in range(num_test):
    bp = EPBP(g, n=10, proposal_approximation='simple')
    start_time = time.process_time()
    bp.run(10, log_enable=False)
    time_cost[name] = (time.process_time() - start_time) / num_test + time_cost.get(name, 0)
    print(name, f'time {time.process_time() - start_time}')
    for i, key in enumerate(key_list):
        res[i, j] = bp.probability(0.8, 1, rvs_table[key])
for i, key in enumerate(key_list):
    ans[key] = np.average(res[i, :])
variance[name] = np.average(np.var(res, axis=1))
print(name, 'var', variance[name])

# save ans
ans_array = list()
for key in key_list:
    ans_array.append((f'{key[0]},{key[1]}', ans[key]))
np.save('Data/smoker_ans', np.array(ans_array))

# load ans
ans_array = np.load('Data/smoker_ans.npy')
for line in ans_array:
    key = tuple([x.strip() for x in line[0].split(',')])
    ans[key] = float(line[1])

name = 'LEPBP'
res = np.zeros((len(key_list), num_test))
for j in range(num_test):
    bp = HybridLBP(g, n=10, proposal_approximation='simple')
    start_time = time.process_time()
    bp.run(10, log_enable=False)
    time_cost[name] = (time.process_time() - start_time) / num_test + time_cost.get(name, 0)
    print(name, f'time {time.process_time() - start_time}')
    for i, key in enumerate(key_list):
        res[i, j] = bp.probability(0.8, 1, rvs_table[key])
variance[name] = np.average(np.var(res, axis=1))
for i, key in enumerate(key_list):
    res[i, :] -= ans[key]
avg_diff[name] = np.average(np.average(abs(res), axis=1))
print(name, 'var', variance[name])
print(name, 'diff', avg_diff[name])

name = 'C2FEPBP'
res = np.zeros((len(key_list), num_test))
for j in range(num_test):
    bp = HybridLBP(g, n=10, proposal_approximation='simple')
    start_time = time.process_time()
    bp.run(10, c2f=True, log_enable=False)
    time_cost[name] = (time.process_time() - start_time) / num_test + time_cost.get(name, 0)
    print(name, f'time {time.process_time() - start_time}')
    for i, key in enumerate(key_list):
        res[i, j] = bp.probability(0.8, 1, rvs_table[key])
variance[name] = np.average(np.var(res, axis=1))
for i, key in enumerate(key_list):
    res[i, :] -= ans[key]
avg_diff[name] = np.average(np.average(abs(res), axis=1))
print(name, 'var', variance[name])
print(name, 'diff', avg_diff[name])

print('######################')
for name, v in time_cost.items():
    print(name, f'avg time {v}')
for name, v in avg_diff.items():
    print(name, f'diff {v}')
for name, v in variance.items():
    print(name, f'var {v}')
