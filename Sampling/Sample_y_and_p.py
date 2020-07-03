import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
import time
import pydde as d
from datetime import date

#Parameters
samplenum = 20000
time_length = 60 #seconds
batch_size = 1000
filenum = int(samplenum/batch_size)

# Generate simulation
dyn = d.PyDyn('../Data/point-mass_pendulum.sim', time_length)
state_init = dyn.compute(dyn.p_init)
f = dyn.f(state_init, dyn.p_init)
df = dyn.df_dp(state_init, dyn.p_init)
dy = dyn.dy_dp(state_init, dyn.p_init)

#Sample targets
y_target = np.zeros((samplenum,3))
y_target[:,0] = np.random.rand(samplenum)*2
y_target[:,1] = 2-np.random.rand(samplenum)*2
#y_target[:,1] = 2
y_target[:,2] = np.random.rand(samplenum)*2

#Sample p
today = date.today()
print('STARTING TO SAMPLE', today)
start_time = time.time()

for b in range(filenum):
    data = {}
    data['parameter'] = []
    data['y_target'] = []
    data['loss'] = []
    data['iterations'] = []
    data['gradnorm'] = []
    for i in range(batch_size):
        params = dyn.get_p(y_target[b*batch_size+i,:], dyn.p_init)
        data['y_target'].append(list(y_target[b*batch_size+i,:]))
        data['parameter'].append(list(params.ptraj))
        data['loss'].append(params.loss)
        data['iterations'].append(params.iterations)
        data['gradnorm'].append(params.grad)

    with open(f'../Data/Samples/data_20k_2x2x2/data_{b}.json', 'w') as outfile:
        json.dump(data, outfile)
    print(f'batch number {b+1} completed')
    print(f'\nDuration: {(time.time() - start_time)/60:.3f} min') # print the time elapsed
    
#add description file
Description = {}
Description['Description'] = [{'samplenum': samplenum, 'samplesperfile': batch_size, 'time length': time_length, 'range from': [0, 2, 0], 'range to': [2, 0, 2]}]
with open('../Data/Samples/data_20k_2x2/Description.json', 'w') as outfile:
    json.dump(Description, outfile)