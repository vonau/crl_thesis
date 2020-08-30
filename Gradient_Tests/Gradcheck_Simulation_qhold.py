####################################
#LIBRARIES

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
import pydde as dde
import json

#Parameters
samplenum = 5
input_size = 9 # q (0:3), qdot (3:6), current p (6:9) 
output_size = 3
nTimeSteps = 10
simulation_file_path = '../Data/Simulations/pm_target.sim'
objective_file_path = f'../Data/Objectives/pm_qhold_{nTimeSteps}.obj'

# set log level
dde.set_log_level(dde.LogLevel.off)

#######################################
# LOAD SIMULATION AND OBJECTIVE FUNCTION
dyn = dde.DynamicSequence()
dyn.loadFile(simulation_file_path, nTimeSteps)
p_init = np.zeros(dyn.p0.size*nTimeSteps)
for i in range(0,nTimeSteps):
	p_init[i*dyn.p0.size : (i+1)*dyn.p0.size] = dyn.p0
state_init = dyn.q(p_init)
r = dyn.r(state_init, p_init)
dr = dyn.dr_dp(state_init, p_init)
dq = dyn.dq_dp(state_init, p_init)

##########################################
#SAMPLE TARGETS
data = np.zeros((samplenum,input_size))
for i in range(input_size):
    data[:,i] = np.random.rand(samplenum)
data[:, 6:9] = data[:, 6:9]*3-1
data[:, 3:6] = data[:, 3:6]-0.5

#############################################
#LOAD OBJECTIVE PYDDE_V2
obj = dde.InverseObjective(dyn)
obj.loadFile(objective_file_path)
objective_json = json.load(open(objective_file_path))
#objective_json["parameter setup"]["nTimeSteps"] = nTimeSteps

#############################################
#GENERATE OPTIMIZATION PYDDE_V2
opt = dde.Newton()

##########################################
#SAMPLE P
for i in range(samplenum):
    p[:,i] = opt.minimize(obj, p_init)
p = torch.tensor(p, requires_grad = True).t().double()
input = p

##########################################
#BUILD CUSTOM SIMULATION FUNCTION
class Simulate(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, data):
        #print(f'input: {input.shape}')
        p = input.clone().numpy().transpose()
        q_pred = torch.ones([len(p[0, :]),3*nTimeSteps])
        for i in range(len(p[0, :])):
            #dyn.q0 = data[i, 0:3]
            #dyn.qdot0 = data[i, 3:6]
            state = dyn.q(p[:,i], data[i, 0:3], data[i, 3:6])
            q_pred[i, :] = torch.tensor(state.q)
        #print(f'q_pred: {q_pred.shape}')
        
        ctx.save_for_backward(input)
        
        return q_pred
        
    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        input, = ctx.saved_tensors
        p = input.clone().numpy().transpose()
        dq_dp_batch = torch.zeros([3*nTimeSteps, 3*nTimeSteps])
        for i in range(len(p[0, :])):
            state = dyn.q(p[:, i])
            dq_dp = dyn.dq_dp(state, p[:, i])
            dq_dp = torch.tensor(dq_dp)
            dq_dp_batch = dq_dp_batch + dq_dp
        #print(f'dq/dp_batch: {dy_dp_batch/samplenum}')
        
        grad_input = grad_output.mm(dq_dp_batch.float()/len(p[0,:]))
        #print(f'shape of grad input: {grad_input.shape}')
        #print(f'shape of grad output: {grad_output.shape}')
        return grad_input, None

Simulate = Simulate.apply

#GRADCHECK
print("Running gradcheck...")
from torch.autograd import gradcheck

test = gradcheck(Simulate, (input,), eps=2e-3, atol=1e-2, raise_exception = True)
print(test)