import numpy as np
from itertools import product


instances = []
data = []

num_people = 10

# create instance
for a in range(num_people):
    instances.append(['people', f'p{a}'])

# save instance
np.save('Data/smoker_instances', np.array(instances))

# generate data
for a in range(num_people):
    for b in range(num_people):
        if a < b:
            data.append([f'friend,p{a},p{b}', np.random.choice([0, 1])])

A = np.random.choice(num_people, 7, replace=False)
for a in A:
    data.append([f'smoke,p{a}', np.random.random_sample()])

for line in data:
    print(line)

# save data
np.save('Data/smoker_data', np.array(data))
