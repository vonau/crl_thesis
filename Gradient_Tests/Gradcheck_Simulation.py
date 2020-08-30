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
samplenum = 2
input_size = 3
output_size = 3
nTimeSteps = 1
simulation_file_path = '../Data/Simulations/pm_target.sim'
objective_file_path = f'../Data/Objectives/pm_target_{nTimeSteps}.obj'

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
"""
    Setting default torch type to double accoding to: https://discuss.pytorch.org/t/how-to-set-dtype-for-nn-layers/31274/4
"""
torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)
# set log level
dde.set_log_level(dde.LogLevel.off)

#######################################
# LOAD SIMULATION AND OBJECTIVE FUNCTION
dyn = dde.DynamicSequence()
dyn.loadFile(simulation_file_path, nTimeSteps)
p_init = np.zeros(dyn.p0.size*nTimeSteps)
for i in range(nTimeSteps):
	p_init[i*dyn.p0.size : (i+1)*dyn.p0.size] = dyn.p0
state_init = dyn.q(p_init)
r = dyn.r(state_init, p_init)
dr = dyn.dr_dp(state_init, p_init)
dq = dyn.dq_dp(state_init, p_init)

#sample only p_init
p = np.ones((3*nTimeSteps, samplenum))
for i in range(samplenum):
    p[:,i] = p_init
p[:,1] = p_init + 0.2
print(p)

##########################################
#SAMPLE TARGETS
y_target = np.zeros((samplenum,3))
y_target[:,2] = np.random.rand(samplenum)
y_target[:,1] = np.random.rand(samplenum)
y_target[:,0] = np.random.rand(samplenum)

print(y_target)

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
# for i in range(samplenum):
#     objective_json["objectives"]["pmTargetPositions"][0]["targetPos"] = ([[y_target[i,0]],[y_target[i,1]],[y_target[i,2]]]) 
#     obj.loadJson(objective_json)
#     p[:,i] = opt.minimize(obj, p_init)
#input = torch.tensor(p, requires_grad = True).t()
input = torch.tensor(p, requires_grad = True)

##########################################
#BUILD CUSTOM SIMULATION FUNCTION
class Simulate(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        #print(f'input: {input.shape}')
        p = input.detach().clone().numpy()
        #print(f'p: {p}')
        q_pred = torch.ones([dyn.nDofs*nTimeSteps, len(p[0, :])])
        for i in range(len(p[0, :])):
            state = dyn.q(p[:,i])
            q_pred[:, i] = torch.tensor(state.q)
        #print(f'q_pred: {q_pred}')
        
        ctx.save_for_backward(input)
        
        return q_pred
        
    @staticmethod
    def backward(ctx, grad_output):
        #print(f'grad_output{grad_output}')
        input, = ctx.saved_tensors
        p = input.detach().clone().numpy()
        dq_dp = np.zeros([dyn.nDofs*nTimeSteps, dyn.nParameters*nTimeSteps])
        for i in range(len(p[0, :])):
            state = dyn.q(p[:, i])
            dq_dp_single = dyn.dq_dp(state, p[:, i])
            #dq_dp_single = torch.tensor(dq_dp_single)
            dq_dp = dq_dp + dq_dp_single
        #print(f'dq/dp: {dq_dp/samplenum}')
        dq_dp = torch.tensor(dq_dp)
        #grad_input = grad_output.mm(torch.div(dq_dp,samplenum)).t()
        grad_input = (dq_dp/samplenum).mm(grad_output)
        #print(f'shape of grad input: {grad_input.shape}')
        #print(f'shape of grad output: {grad_output.shape}')
        return grad_input

Simulate = Simulate.apply

#GRADCHECK
print("Running gradcheck...")
from torch.autograd import gradcheck

test = gradcheck(Simulate, (input,), eps=1e-8, atol=1e-8, raise_exception = True)
print(test)