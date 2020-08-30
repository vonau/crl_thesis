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
nTimeSteps = 60 #at 60Hz
batch_size = 1000 # amount of samples per file
input_size = 15 # target (0:3), q (3:6), qdot (6:9), qddot (9:12), p_now (12:15) 
samples_per_sim = 1 # amount of states taken per generated trajectory
threshhold2 = 0.1
use_case = 'varstart'
filenum = int(samplenum/batch_size)
sample_file_path = f'../Data/Samples/data_varstart_{nTimeSteps}tsteps' + now.strftime("%H%M") + '/'
simulation_file_path = "../Data/Simulations/pm_target.sim"
objective_file_path = f'../Data/Objectives/pm_target.obj'

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

#SAMPLE TARGETS
print("SAMPLING STARTED...")
tot_simulations_needed = np.ceil(samplenum/samples_per_sim).astype(int)
print(tot_simulations_needed)

input = np.zeros((samplenum,input_size))
targets = np.zeros((tot_simulations_needed,3))
targets[:,0] = np.random.rand(tot_simulations_needed)
targets[:,1] = np.random.rand(tot_simulations_needed)
targets[:,2] = np.random.rand(tot_simulations_needed)
#targets[:,2] = np.random.rand(tot_simulations_needed)/2

############################################
#LOAD SIMULATION PYDDE_V2
dyn = dde.DynamicSequence()
dyn.loadFile(simulation_file_path, nTimeSteps)
p_init_0 = np.zeros(dyn.p0.size*nTimeSteps)
for i in range(nTimeSteps):
	p_init_0[i*dyn.p0.size : (i+1)*dyn.p0.size] = dyn.p0
state_init = dyn.q(p_init_0)
r = dyn.r(state_init, p_init_0)
dr = dyn.dr_dp(state_init, p_init_0)
dq = dyn.dq_dp(state_init, p_init_0)

#############################################
#GENERATE OPTIMIZATION PYDDE_V2
opt = dde.Newton()
opt.maxIterations = 1

####################################
#SAMPLE REAL STATES
# target (0:3), q (3:6), qdot (6:9), qddot (9:12), p_now (12:15) 
p_init = p_init_0
start_time = time.time()
print("SAMPLING INITIAL STATES...")
#define json objects
dyn_json = json.load(open(simulation_file_path))
objective_json = json.load(open(objective_file_path))
for i_ in range(tot_simulations_needed):
    #load objective
    obj = dde.InverseObjective(dyn)
    objective_json["objectives"]["pmTargetPositions"][0]["timeIndex"] = nTimeSteps-1
    objective_json["objectives"]["pmTargetPositions"][0]["targetPos"][0] = [targets[i_,0]]
    objective_json["objectives"]["pmTargetPositions"][0]["targetPos"][1] = [targets[i_,1]]
    objective_json["objectives"]["pmTargetPositions"][0]["targetPos"][2] = [targets[i_,2]]
    obj.loadJson(objective_json)
    #Trajectory Optimization
    g = 0
    u =0
    u2=0
    residual = 1
    while residual > 0.1:
        p_i = opt.minimize(obj, p_init)
        p_init = p_i
        #print(f'iteration{g} with residual {opt.lastResidual}')
        g = g+1
        if g == 300:
            #change starting position to initial position
            u=u+1
            dyn = dde.DynamicSequence()
            dyn.loadFile(simulation_file_path, nTimeSteps)
            p_init = p_init_0
            obj = dde.InverseObjective(dyn)
            objective_json["objectives"]["pmTargetPositions"][0]["timeIndex"] = nTimeSteps-1
            objective_json["objectives"]["pmTargetPositions"][0]["targetPos"][0] = [targets[i_,0]]
            objective_json["objectives"]["pmTargetPositions"][0]["targetPos"][1] = [targets[i_,1]]
            objective_json["objectives"]["pmTargetPositions"][0]["targetPos"][2] = [targets[i_,2]]           
            obj.loadJson(objective_json)
            for i in range(nTimeSteps):
	            p_init[i_*dyn.p0.size : (i_+1)*dyn.p0.size] = dyn.p0
        residual = obj.evaluate(p_i)
        if g == 400:
            u2=u2+1
            residual = residual-1
        
    state_i = dyn.q(p_i)
    n_q = np.floor(dyn.nDofs*nTimeSteps/samples_per_sim).astype(int)
    n_p = np.floor(dyn.nParameters*nTimeSteps/samples_per_sim).astype(int)
    
    for s in range(samples_per_sim):
        s_ = s+1
        input[i_*samples_per_sim+s,3:6] = state_i.q[n_q*s_-dyn.nDofs:n_q*s_]
        input[i_*samples_per_sim+s,6:9] = state_i.qdot[n_q*s_-dyn.nDofs:n_q*s_]
        input[i_*samples_per_sim+s,9:12] = state_i.qddot[n_q*s_-dyn.nDofs:n_q*s_]
        input[i_*samples_per_sim+s,12:15] = p_i[n_p*s_-dyn.nParameters:n_p*s_]

    for l in range(dyn.nDofs):
        dyn_json["simulation"]["q"][l] = [input[i_*samples_per_sim+s,3+l]]
        dyn_json["simulation"]["qdot"][l] = [input[i_*samples_per_sim+s,6+l]]
        dyn_json["simulation"]["qddot"][l] = [input[i_*samples_per_sim+s,9+l]]
    for l in range(dyn.nParameters):
        dyn_json["simulation"]["p"][l] = [input[i_*samples_per_sim+s,12+l]]
    dyn.loadJson(dyn_json, nTimeSteps)
    dyn.p0 = input[i_*samples_per_sim,12:15]
    for s_1 in range(nTimeSteps):
        p_init[s_1*dyn.p0.size : (s_1+1)*dyn.p0.size] = dyn.p0
    if i_%(tot_simulations_needed/10) == 0:
        print(f'{(time.time() - start_time)/60} simulation number {i_} complete')
print(f'Jumped to initial state due to high error {u} times')
print(f'Terminated off optimization due to high error {u2} times')
print(f'TOTAL TIME TO SAMPLE STARTIG STATES: {(time.time() - start_time)/60}')

#SAMPLE TARGETS
print("SAMPLING NEW TARGETS...")
print(f'\nDuration to start: {(time.time() - start_time)/60:.3f} min')
input[:,0] = np.random.rand(samplenum)
input[:,1] = np.random.rand(samplenum)
input[:,2] = np.random.rand(samplenum)
p_init = p_init_0
c= 0

#############################################
print("SAMPLING CONTROLS...")

for b in range(filenum):
    data = {}
    data['p'] = []
    data['q_target'] = []
    data['q'] = []
    data['qdot'] = []
    data['qddot'] = []
    data['p_now'] = []
    data['loss'] = []
    for i_2 in range(batch_size):
        #Change DynamicSequence
        dyn_json = json.load(open(simulation_file_path))
        for l in range(dyn.nDofs):
            dyn_json["simulation"]["q"][l] = [input[b*batch_size+i_2,3+l]] 
            dyn_json["simulation"]["qdot"][l] = [input[b*batch_size+i_2,6+l]] 
            dyn_json["simulation"]["qddot"][l] = [input[b*batch_size+i_2,9+l]] 
        for l in range(dyn.nParameters):
            dyn_json["simulation"]["p"][l] = [input[b*batch_size+i_2,12+l]] 
        dyn.loadJson(dyn_json, nTimeSteps)
        dyn.p0 = input[b*batch_size+i,12:15]
        for s in range(nTimeSteps):
            p_init[s*dyn.p0.size : (s+1)*dyn.p0.size] = dyn.p0
        #Change Objective
        obj = dde.InverseObjective(dyn)
        objective_json = json.load(open(objective_file_path))
        objective_json["objectives"]["pmTargetPositions"][0]["timeIndex"] = nTimeSteps-1
        objective_json["objectives"]["pmTargetPositions"][0]["targetPos"] = ([[input[b*batch_size+i_2,0]],[input[b*batch_size+i_2,1]],[input[b*batch_size+i_2,2]]]) 
        obj.loadJson(objective_json)

        #Trajectory Optimization
        g = 0
        residual = 1
        while residual > threshhold2:
            p_i = opt.minimize(obj, p_init)
            p_init = p_i
            #print(f'iteration{g} with residual {opt.lastResidual}')
            g = g+1
            residual = obj.evaluate(p_i)
            if g == 500:
                c = c+1
                residual = 0
                input[b*batch_size+i_2,:]=0
                break
         
        data['q_target'].append(list(input[b*batch_size+i_2,0:3]))
        data['q'].append(list(input[b*batch_size+i_2,3:6]))
        data['qdot'].append(list(input[b*batch_size+i_2, 6:9]))
        data['qddot'].append(list(input[b*batch_size+i_2,9:12]))
        data['p_now'].append(list(input[b*batch_size+i_2,12:15]))
        data['p'].append(list(p_i))
        data['loss'].append(residual)
        #data['iterations'].append(params.iterations)
        #data['gradnorm'].append(params.grad)

    with open(sample_file_path + f'data_{b}.json', 'w') as outfile:
        json.dump(data, outfile)
    print(f'batch number {b+1} completed, jumped to initial state {c} times')
    print(f'\nDuration: {(time.time() - start_time)/60:.3f} min') # print the time elapsed
    
#add description file
Description = {}
Description['Description'] = [{'samplenum': samplenum, 'samplesperfile': batch_size, 'time length': nTimeSteps, 'sample space': "x-, z- and y-direction 0 to 1"}]
with open(sample_file_path + 'Description.json', 'w') as outfile:
    json.dump(Description, outfile)
print('SAMPLING COMPLETE.')
print(f'Jumped to initial state due to high error {c} times')
print(f'TOTAL TIME TO SAMPLE: {(time.time() - start_time)/60}')