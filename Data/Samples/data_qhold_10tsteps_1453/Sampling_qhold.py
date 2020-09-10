#LIBRARIES
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
import time
from datetime import datetime
import pydde as dde
import os

#PARAMETERS
now = datetime.now()
samplenum = 20000
nTimeSteps = 10 #at 60Hz
batch_size = 1000
input_size = 9 # q (0:3), qdot (3:6), current p (9:12) 
#samples_per_sim = int(nTimeSteps/6) # amount of states taken per generated trajectory
samples_per_sim = 4
filenum = int(samplenum/batch_size)
use_case = 'qhold'
sample_file_path = f'../Data/Samples/data_qhold_{nTimeSteps}tsteps_' + now.strftime("%H%M") + '/'
simulation_file_path = "../Data/Simulations/pm_target.sim"
objective_file_path = f'../Data/Objectives/pm_qhold.obj'
objective_file_path_sampling = f'../Data/Objectives/pm_target.obj'

# check dde version
print("using dde version: " + dde.__version__)
# set log level
dde.set_log_level(dde.LogLevel.off)
print(f'log level set to {dde.get_log_level()}')

# CREATE DIRECTORY FOR SAMPLES
try:
    os.mkdir(sample_file_path)
except OSError:
    print ("Creation of the directory %s failed" % sample_file_path)
else:
    print ("Successfully created the directory %s" % sample_file_path)

print("SAMPLING STARTED...")

############################################
#LOAD SIMULATION PYDDE_V2
dyn = dde.DynamicSequence()
dyn.loadFile(simulation_file_path, nTimeSteps)
p_init_0 = np.zeros(dyn.p0.size*nTimeSteps)
for i in range(nTimeSteps):
	p_init_0[i*dyn.p0.size : (i+1)*dyn.p0.size] = dyn.p0
state_init = dyn.q(p_init_0)
q_0 = dyn.q0
qdot_0 = dyn.qdot0
qddot_0 = dyn.qddot0
r = dyn.r(state_init, p_init_0)
dr = dyn.dr_dp(state_init, p_init_0)
dq = dyn.dq_dp(state_init, p_init_0)

#############################################
#GENERATE OPTIMIZATION
opt = dde.Newton()
#opt.maxIterations = 800

#OBJECTIVE
obj = dde.InverseObjective(dyn)
objective_json = json.load(open(objective_file_path))
obj.loadJson(objective_json)

####################################
#SAMPLE STATES
# q (0:3), qdot (3:6), qddot(6:9), current p (9:12) 
p_init = p_init_0
start_time = time.time()
print("SAMPLING INITIAL STATES...")
dyn_json = json.load(open(simulation_file_path))
objective_json = json.load(open(objective_file_path_sampling))

input = np.zeros((samplenum,input_size))
# sample q
input[:,0] = np.random.rand(samplenum)
input[:,1] = np.random.rand(samplenum)
input[:,2] = np.random.rand(samplenum)
# sample qdot
input[:,3] = np.random.rand(samplenum)-0.5
input[:,4] = np.random.rand(samplenum)-0.5
input[:,5] = np.random.rand(samplenum)-0.5
# sample p_now
input[:,6] = np.random.rand(samplenum)*3-1
input[:,7] = np.random.rand(samplenum)*3-1
input[:,8] = np.random.rand(samplenum)*3-1
print(f'TOTAL TIME TO SAMPLE STARTIG STATES: {(time.time() - start_time)/60}')

###############################################
#SAMPLE CONTROLS
c=0 # counter for how often the error is too high

for b in range(filenum):
    data = {}
    data['p'] = []
    data['q_target'] = []
    data['q'] = []
    data['qdot'] = []
    #data['qddot'] = []
    data['p_now'] = []
    data['loss'] = []
    data['iterations'] = []
    data['lineSearchIterations'] = []

    for i in range(batch_size):
        #Change DynamicSequence
        for l in range(dyn.nDofs):
            dyn_json["simulation"]["q"][l] = [input[b*batch_size+i,0+l]] 
            dyn_json["simulation"]["qdot"][l] = [input[b*batch_size+i,3+l]] 
            #dyn_json["simulation"]["qddot"][l] = [input[b*batch_size+i,6+l]] 
        for l in range(dyn.nParameters):
            dyn_json["simulation"]["p"][l] = [input[b*batch_size+i,6+l]] 
        dyn.loadJson(dyn_json, nTimeSteps)
        dyn.p0 = input[b*batch_size+i,6:9]
        for s in range(nTimeSteps):
            p_init[s*dyn.p0.size : (s+1)*dyn.p0.size] = dyn.p0

        #Trajectory Optimization
        # g = 0
        # residual = 1
        # while residual > 0.001:
        p_i = opt.minimize(obj, p_init)
            # p_init = p_i
            # #print(f'iteration{g} with residual {opt.lastResidual}')
            # g = g+1
                
            # if g == 3:
            #     c = c+1
            #     break
            # residual = obj.evaluate(p_i)

        #Store the Data
        data['q'].append(list(input[b*batch_size+i,0:3]))
        data['qdot'].append(list(input[b*batch_size+i,3:6]))
        #data['qddot'].append(list(input[b*batch_size+i,6:9]))
        data['p_now'].append(list(input[b*batch_size+i,6:9]))
        data['p'].append(list(p_i))
        data['loss'].append(obj.evaluate(p_i))
        data['iterations'].append(opt.getLastIterations())
        data['lineSearchIterations'].append(opt.lastLineSearchIterations)

    with open(sample_file_path + f'data_{b}.json', 'w') as outfile:
        json.dump(data, outfile)
    print(f'batch number {b+1} completed')
    print(f'\nDuration: {(time.time() - start_time)/60:.3f} min') # print the time elapsed
    
#add description file
Description = {}
Description['Description'] = [{'samplenum': samplenum, 'samplesperfile': batch_size, 'time length': nTimeSteps, 'sample space': "x-, y- and z direction 0 to 1"}]
with open(sample_file_path + 'Description.json', 'w') as outfile:
    json.dump(Description, outfile)
print('SAMPLING COMPLETE.')
print(f'Jumped to initial state due to high error {c} times')
print(f'TOTAL TIME TO SAMPLE: {(time.time() - start_time)/60}')