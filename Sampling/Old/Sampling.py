# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
import time
import pydde as d


# %%
#Parameters
samplenum = 20000
time_length = 60 #seconds
batch_size = 500
filenum = int(samplenum/batch_size)


# %%
# Generate simulation
dyn = d.PyDyn('../Data/point-mass_pendulum.sim', time_length)
state_init = dyn.compute(dyn.p_init)
f = dyn.f(state_init, dyn.p_init)
df = dyn.df_dp(state_init, dyn.p_init)
dy = dyn.dy_dp(state_init, dyn.p_init)


# %%
#Sample targets
y_target = np.zeros((samplenum,3))
y_target[:,0] = np.random.rand(samplenum)*2
y_target[:,1] = 2-np.random.rand(samplenum)*2
#y_target[:,1] = 2
y_target[:,2] = np.random.rand(samplenum)*2
y_target[20:50, :]
y_target[:, 0].size


# %%
#Sample p
for b in range(filenum):
    data = {}
    data['parameter'] = []
    data['y_target'] = []
    start_time1 = time.time()
    for i in range(batch_size):
        data['y_target'].append(list(y_target[i,:]))
        data['parameter'].append(list(dyn.get_p(y_target[i,:], dyn.p_init)))

    print(f'batch number {b} completed')
    with open(f'../Data/Samples/data_20k_2x2/data_{b}.json', 'w') as outfile:
        json.dump(data, outfile)

print(f'\nDuration: {(time.time() - start_time1)/60:.3f} min') # print the time elapsed
#add description file
Description = {}
Description['Description'] = [{'samplenum': samplenum, 'samplesperfile': batch_size, 'time length': time_length, 'range from': [0, 2, 0], 'range to': [2, 0, 2]}]
with open('../Data/Samples/data_20k_2x2/Description.json', 'w') as outfile:
    json.dump(Description, outfile)

