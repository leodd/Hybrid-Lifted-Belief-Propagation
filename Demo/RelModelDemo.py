from RelationalGraph import *
from Potential import GaussianPotential
from HybridLBPLogVersion import HybridLBP
from EPBPLogVersion import EPBP
from GaBP import GaBP
import numpy as np
import time


X = np.load('Data/rel_model_instances.npy')
D = np.load('Data/rel_model_data.npy')

instance_category = []
instance_bank = []
for row in X:
    if row[0] == 'category':
        instance_category.append(row[1])
    else:
        instance_bank.append(row[1])

data = dict()
for row in D:
    key = tuple([x.strip() for x in row[0].split(',')])
    if key[0] == 'revenue': continue
    data[key] = float(row[1])
data[('recession', 'all')] = 50

domain_percentage = Domain((-100, 100), continuous=True, integral_points=linspace(-100, 100, 30))
domain_billion = Domain((-50, 50), continuous=True, integral_points=linspace(-50, 50, 30))

p1 = GaussianPotential([0., 0.], [[10., -7.], [-7., 10.]])
p2 = GaussianPotential([0., 0.], [[10., 5.], [5., 10.]])
p3 = GaussianPotential([0., 0.], [[10., 7.], [7., 10.]])

lv_recession = LV(('all',))
lv_category = LV(instance_category[:100])
lv_bank = LV(instance_bank[:10])

atom_recession = Atom(domain_percentage, logical_variables=(lv_recession,), name='recession')
atom_market = Atom(domain_percentage, logical_variables=(lv_category,), name='market')
atom_loss = Atom(domain_billion, logical_variables=(lv_category, lv_bank), name='loss')
atom_revenue = Atom(domain_billion, logical_variables=(lv_bank,), name='revenue')

f1 = ParamF(p1, nb=(atom_recession, atom_market))
f2 = ParamF(p2, nb=(atom_market, atom_loss))
f3 = ParamF(p3, nb=(atom_loss, atom_revenue))

rel_g = RelationalGraph()
rel_g.atoms = (atom_recession, atom_revenue, atom_loss, atom_market)
rel_g.param_factors = (f1, f2, f3)
rel_g.data = data

rel_g.init_nb()
g, rvs_table = rel_g.grounded_graph()
print('number of vr', len(g.rvs))
num_evidence = 0
for rv in g.rvs:
    if rv.value is not None:
        num_evidence += 1
print('number of evidence', num_evidence)

key_table = []
j = 0
for key in rvs_table:
    key_table.append(key)
    j += 1
num_test = 2
result_table = np.zeros((len(rvs_table), num_test))
time_table = []

for i in range(num_test):
    bp = HybridLBP(g, n=20)
    start_time = time.process_time()
    bp.run(15, log_enable=False)
    time_table.append(time.process_time() - start_time)

    j = 0
    for key, rv in rvs_table.items():
        result_table[j, i] = bp.map(rv)
        j += 1

print('average time', np.mean(time_table))

# for i in range(len(rvs_table)):
#     key = key_table[i]
#     mean = np.mean(result_table[i])
#     variance = np.var(result_table[i])
#     print(key, mean, variance)

bp = GaBP(g)
bp.run(20, log_enable=False)

# for key, rv in rvs_table.items():
#     print(key, bp.map(rv))

i = 0
err_history = list()
for key, rv in rvs_table.items():
    ans = bp.map(rv)
    err = result_table[i] - ans
    mean_l1_err = np.average(abs(err))
    err_history.append(mean_l1_err)
    print(key, np.average(result_table[i]), mean_l1_err)
    i += 1
print('average err', np.average(err_history))
print('max err', np.max(err_history))
