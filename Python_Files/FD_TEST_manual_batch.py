####################################
#LIBRARIES

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
import pydde as d
import matplotlib.pyplot as plt

####################################
#Parameters
samplenum = 5
input_size = 3
output_size = 3
time_length = 3; #seconds

# Generate simulation
dyn = d.PyDyn('../Data/point-mass_pendulum.sim', time_length)
state_init = dyn.compute(dyn.p_init)
f = dyn.f(state_init, dyn.p_init)
df = dyn.df_dp(state_init, dyn.p_init)
dy = dyn.dy_dp(state_init, dyn.p_init)

#Sample targets only variables in z direction
y_target = np.zeros((samplenum, 3))
y_target[:,2] = np.random.rand(samplenum)
y_target[:,1] = 2

###################################
#USE P FROM TRAJ_OPT
p = np.ones((3*time_length, samplenum))
for i in range(samplenum):
    p[:,i] = dyn.get_p(y_target.transpose()[:,i], dyn.p_init)
p = torch.tensor(p, requires_grad = True).t()

##############################
#USE P_INIT ONLY
# p = np.ones((3*time_length, samplenum))
# for i in range(samplenum):
#     p[:,i] = dyn.p_init
# p = torch.tensor(p, requires_grad = True).t()
# input = p.double()
# print(p)
# print(p.shape)

################################
#BUILD CUSTOM FUNCTION

class Simulate(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        #print(f'input: {input.shape}')
        p = input.clone().numpy().transpose()
        y_pred = torch.ones([len(p[0, :]),3])
        for i in range(len(p[0, :])):
            state = dyn.compute(p[:,i])
            y_pred[i, :] = torch.tensor(state.y[-3:])
        #print(f'y_pred: {y_pred.shape}')
        
        ctx.save_for_backward(input)
        
        return y_pred
        
    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        input, = ctx.saved_tensors
        p = input.clone().numpy().transpose()
        dy_dp_batch = torch.zeros([3, 3*time_length])
        for i in range(len(p[0, :])):
            state= dyn.compute(p[:, i])
            dy_dp = dyn.dy_dp(state, p[:, i])
            dy_dp = torch.tensor(dy_dp[-3:, :])
            dy_dp_batch = dy_dp_batch + dy_dp
        #print(f'dy/dp_batch: {dy_dp_batch/samplenum}')
        
        grad_input = torch.tensor(grad_output.double().mm(dy_dp_batch/samplenum))
        #print(f'shape of grad input: {grad_input.shape}')
        #print(f'shape of grad output: {grad_output.shape}')
        return grad_input

Simulate = Simulate.apply

################################
#GET ANALYTICAL GRADIENT

# Error for whole simulation
from numpy import linalg as LA

p = torch.tensor(p, requires_grad = True).double()
y = Simulate(p)
grad_output = torch.ones([samplenum,3]).double()
y.backward(grad_output)
dy_dp = p.grad.double()
#print(f'dy_dp = {dy_dp}')

################################
#GET NUMERICAL GRADIENT WITH DIFFERENT PERTUBATIONS

dy_dp_s = np.zeros((3,len(dyn.p_init)))
#dy_dp_FD = np.zeros((3,len(dyn.p_init)))
#FD = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
FD = np.logspace(np.log10(0.000001),np.log10(0.01),num = 100)
Grads = {}
Err = []
analytical_grad = dy_dp.detach().numpy()

for f, fd in enumerate(FD):
    dy_dp_FD = np.zeros((3,len(dyn.p_init)))
    for s in range(samplenum):
        for i in range(len(dyn.p_init)):
            dp= np.zeros(len(dyn.p_init))
            dp[i] = fd
            dp = torch.tensor(dp)
            p_s = p[s,:]
            p_s = p_s.unsqueeze(0)
            y_p = Simulate(p_s + dp)
            y_m = Simulate(p_s - dp)
            y_p = y_p.detach().numpy()
            y_m = y_m.detach().numpy()
            grad_i = (y_p - y_m) / (2 * fd)
            dy_dp_s[:, i]= grad_i
        dy_dp_FD = dy_dp_FD + dy_dp_s
    dy_dp_FD_ten = torch.tensor(dy_dp_FD)
    Grads[fd] = torch.tensor(grad_output.double().mm(dy_dp_FD_ten/samplenum))
    #print(f' dy_dp_FD at eps {fd} = \n{dy_dp_FD}')
    Err.append(LA.norm(Grads[fd] - analytical_grad))
#print(Err)

################################
#PLOT ERRORS OVER DIFFERENT PERTURBATIONS

plt.plot(FD, Err)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('error')
plt.title('Errors from FD test for Custom Simulation Function')
plt.xlabel('perturbations for finite differences')
plt.show()

################################
#FD TEST FOR DY/DP FUNCTION

from numpy import linalg as LA
#FD = [1e-2, 1e-3, 1e-4, 1e-5]
#FD = np.linspace(0.000001,0.01, num = 100)
FD = np.logspace(np.log10(0.000001),np.log10(0.01),num = 100)
FDs = []

#Calculate dy_dp with FE
dy_dp = dyn.dy_dp(state_init, dyn.p_init)[-3:, :]
dy_dp_FD = np.zeros((3,len(dyn.p_init)))

for i, fd in enumerate(FD):
    for i in range(len(dyn.p_init)):
        dp= np.zeros(len(dyn.p_init))
        dp[i] = fd
        y_p = dyn.compute(dyn.p_init + dp)
        y_m = dyn.compute(dyn.p_init - dp)
        dy_dp_FD[:, i] = (y_p.y[-3:] - y_m.y[-3:]) / (2* fd)
    err = LA.norm(dy_dp_FD - dy_dp)
    #print(err)
    FDs.append(err)

plt.plot(FD, FDs)
plt.xscale('log')
plt.ylabel('error')
plt.title('Errors from FD test for dy/dp from dde')
plt.xlabel('perturbations for finite differences')
plt.show()