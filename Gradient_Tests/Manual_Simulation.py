########################################
#LIBRARIES
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
import pydde as dde
import matplotlib.pyplot as plt
import json
import os
import time

########################################
#PARAMETERS
nTimeSteps = 30 #at 60 Hz
samplenum = 50
input_size = 3
output_size = 3*nTimeSteps
simulation_file_path = '../Data/Simulations/pointmass.sim'
objective_file_path = f'../Data/Objectives/pm_target_{nTimeSteps}.obj'

# check dde version
print("using dde version: " + dde.__version__)
# set log level
dde.set_log_level(dde.LogLevel.off)
print(f'log level set to {dde.get_log_level()}')

torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=8)

#######################################
# LOAD SIMULATION AND OBJECTIVE FUNCTION
dyn = dde.DynamicSequence()
dyn.loadFile(simulation_file_path, nTimeSteps)
p_init = np.zeros(dyn.p0.size*nTimeSteps)
for i in range(nTimeSteps):
	p_init[i*dyn.p0.size : (i+1)*dyn.p0.size] = dyn.p0

#sample only p_init
p = np.ones((3*nTimeSteps, samplenum))
for i in range(samplenum):
    p[:,i] = p_init

# Objective Function
obj = dde.InverseObjective(dyn)
obj.loadFile(objective_file_path)
objective_json = json.load(open(objective_file_path))
opt = dde.Newton()

##########################################
#SAMPLE TARGETS
y_target = np.zeros((samplenum,3))
y_target[:,2] = np.random.rand(samplenum)
y_target[:,1] = np.random.rand( samplenum)
y_target[:,0] = np.random.rand(samplenum)

###################################
#USE P FROM TRAJ_OPT
p = np.ones((3*nTimeSteps, samplenum))
for i in range(samplenum):
    objective_json["objectives"]["pmTargetPositions"][0]["targetPos"] = ([[y_target[i,0]],[y_target[i,1]],[y_target[i,2]]]) 
    obj.loadJson(objective_json)
    p[:,i] = opt.minimize(obj, p_init)
print(opt)
print(p)

##########################################
#BUILD CUSTOM SIMULATION FUNCTION
class Simulate(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        #print(f'input: {input.shape}')
        p = input.detach().clone().numpy()
        q_pred = torch.ones([len(p[0, :]),dyn.nDofs*nTimeSteps])
        #q_pred = torch.ones([dyn.nDofs*nTimeSteps, len(p[0, :])])
        for i in range(len(p[0, :])):
            state = dyn.q(p[:,i])
            q_pred[i, :] = torch.tensor(state.q)
            #q_pred[:, i] = torch.tensor(state.q)
        #print(f'y_pred: {q_pred.shape}')
        
        ctx.save_for_backward(input)
        
        return q_pred
        
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        p = input.detach().clone().numpy()
        dq_dp_batch = torch.zeros([dyn.nDofs*nTimeSteps, dyn.nParameters*nTimeSteps])
        for i in range(len(p[0, :])):
            state = dyn.q(p[:, i])
            dq_dp = dyn.dq_dp(state, p[:, i])
            dq_dp = torch.tensor(dq_dp)
            dq_dp_batch = dq_dp_batch + dq_dp
        # print(f'dy/dp_batch: {dq_dp_batch/len(p[0, :])}')
        
        #grad_input = grad_output.mm(dq_dp_batch/len(p[0,:]))
        #grad_input = (dq_dp_batch/len(p[0,:])).mm(grad_output)
        grad_input = grad_output.mm(dq_dp_batch/len(p[0,:])).t()
        # print(f'shape of dq/dp: {dq_dp_batch.shape}')
        # print(f'shape of grad input: {grad_input.shape}')
        # print(f'shape of grad output: {grad_output.shape}')
        return grad_input

Simulate = Simulate.apply

################################
#GET ANALYTICAL GRADIENT
print("GETTING ANALYTICAL GRADIENT...")
# Error for whole simulation
from numpy import linalg as LA

p = torch.tensor(p, requires_grad = True)
q = Simulate(p)
grad_output = torch.ones([samplenum,dyn.nDofs*nTimeSteps])
#print(grad_output)
q.backward(grad_output, retain_graph = True)
print(p.grad.shape)
dq_dp = p.grad
analytical_grad = dq_dp.clone().detach().numpy()
#print(f'dy_dp shape = {dy_dp.shape}')
#print(f'dy_dp = {dy_dp}')

################################
#GET NUMERICAL GRADIENT WITH DIFFERENT PERTUBATIONS
print("GETTING NUMERICAL GRADIENT...")
dq_dp_s = np.zeros((len(p_init),len(p_init)))
#dy_dp_FD = np.zeros((3,len(dyn.p_init)))
FD = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9, 5e-10, 1e-10]
d = 1e-8
#FD = [d]
#FD = np.logspace(np.log10(0.00000001),np.log10(0.001),num = 50)
Grads = {}
Err = []
for f, h in enumerate(FD):
    dq_dp_FD = np.zeros((len(p_init),len(p_init)))
    grad_i = np.zeros((len(p_init),len(p_init)))
    for s in range(samplenum):
        for i in range(nTimeSteps*dyn.nParameters):
            pp= p.clone().detach()
            pm= p.clone().detach()
            pp[i,s] = pp[i,s] + h
            pm[i,s] = pm[i,s] - h
            yp = Simulate(pp)
            ym = Simulate(pm)
            yp = yp.clone().detach().numpy()
            ym = ym.clone().detach().numpy()
            # print("ym")
            # print(ym)
            # print("yp")
            # print(yp)
            grad_i[:,i] = (yp[s,:] - ym[s, :]) / (2 * h)
        dq_dp_FD = dq_dp_FD + grad_i
    dq_dp_FD_ten = torch.tensor(dq_dp_FD)
   # print(f'FD_grad: {dq_dp_FD_ten}')

    Grads[f] = grad_output.mm(dq_dp_FD_ten/samplenum).t()
    numerical_grad = grad_output.mm(dq_dp_FD_ten/samplenum).t()
    print(f'p :{p}')
    # print("ANAL_GRAD:")
    # print(analytical_grad)
    # print("NUM_GRAD:")
    # print(numerical_grad)
    #print(f' dy_dp_FD at eps {fd} = \n{dy_dp_FD}')
    Err.append(LA.norm(Grads[f] - analytical_grad))
    Error = LA.norm(Grads[f] - analytical_grad)
print(Err)
    
################################
#PLOT ERRORS OVER DIFFERENT PERTURBATIONS

plt.plot(FD, Err)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('error')
plt.xlabel('perturbations for finite differences')
plt.title(f'Errors from FD-Test for custom simulation function ({samplenum} Samples, {nTimeSteps} timesteps)')
plt.show()
print("Success")