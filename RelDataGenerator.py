import numpy as np
from itertools import product


instances = []
data = []

# create instance
for i in range(200):
    instances.append(['category', f'c{i}'])

for i in range(10):
    instances.append(['bank', f'b{i}'])

# save instance
np.save('Data/reL_model_instances', np.array(instances))

# generate data
idx_category = np.random.choice(200, 50, replace=False)
for i in idx_category[:20]:
    data.append([f'market,c{i}', np.clip(np.random.normal(-50, 20), -100, 100)])
for i in idx_category[30:]:
    data.append([f'market,c{i}', np.clip(np.random.normal(30, 20), -100, 100)])

idx_category = np.random.choice(idx_category, 10, replace=False)
idx_bank = np.random.choice(10, 4, replace=False)
for i, j in product(idx_category, idx_bank):
    data.append([f'loss,c{i},b{j}', np.clip(np.random.normal(0, 20), -50, 50)])

idx_bank = np.random.choice(10, 6, replace=False)
for i in idx_bank:
    data.append([f'revenue,b{i}', np.clip(np.random.normal(0, 20), -50, 50)])

# save data
np.save('Data/reL_model_data', np.array(data))
