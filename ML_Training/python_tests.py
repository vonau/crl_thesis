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
samplenum = 1
input_size = 3
output_size = 3
nTimeSteps = 10
simulation_file_path = '../Data/Simulations/pm_target.sim'
objective_file_path = f'../Data/Objectives/pm_target_{nTimeSteps}.obj'

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

#sample p
p = np.ones((3*nTimeSteps, samplenum))
for i in range(samplenum):
    p[:,i] = p_init
#p = torch.tensor(p, requires_grad = True).t()

##########################################
#SAMPLE TARGETS
y_target = np.zeros((samplenum,3))
y_target[:,2] = np.random.rand(samplenum)
y_target[:,1] = np.random.rand( samplenum)
y_target[:,0] = np.random.rand(samplenum)

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
    objective_json["objectives"]["pmTargetPositions"][0]["targetPos"] = ([[y_target[i,0]],[y_target[i,1]],[y_target[i,2]]]) 
    obj.loadJson(objective_json)
    p[:,i] = opt.minimize(obj, p_init)
p = torch.tensor(p, requires_grad = True).t().double()
input = p

##########################################
#BUILD CUSTOM SIMULATION FUNCTION
class Simulate(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        #print(f'input: {input.shape}')
        p = input.clone().numpy().transpose()
        y_pred = torch.ones([len(p[0, :]),3*nTimeSteps])
        for i in range(len(p[0, :])):
            state = dyn.q(p[:,i])
            y_pred[i, :] = torch.tensor(state.q)
        #print(f'y_pred: {y_pred.shape}')
        
        ctx.save_for_backward(input)
        
        return y_pred
        
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
        #print(f'dy/dp_batch: {dy_dp_batch/samplenum}')
        
        grad_input = grad_output.double().mm(dq_dp_batch/len(p[0,:]))
        #print(f'shape of grad input: {grad_input.shape}')
        #print(f'shape of grad output: {grad_output.shape}')
        return grad_input

Simulate = Simulate.apply

#GRADCHECK
print("Running gradcheck...")
from torch.autograd import gradcheck

test = gradcheck(Simulate, (input,), eps=2e-3, atol=1e-2, raise_exception = True)
print(test)