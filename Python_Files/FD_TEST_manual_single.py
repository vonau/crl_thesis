import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
import pydde as d

#Parameters
samplenum = 1
epochs = 200
input_size = 3
output_size = 3
learning_rate = 0.01
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

p = dyn.get_p(y_target.transpose(), dyn.p_init)
y_target= torch.tensor(y_target, requires_grad= True)

#Building the custon Simulation Activation Function
class Simulate(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        p = input.clone().numpy().transpose()
        state = dyn.compute(p)
        y_pred = torch.tensor(state.y[-3:], requires_grad = True)
        
        ctx.save_for_backward(input)
        
        return y_pred, input
    
    @staticmethod
    def backward(ctx, grad_output, input):
        #print(grad_output.shape)
        input, = ctx.saved_tensors
        p = input.clone().numpy().transpose()
        state= dyn.compute(p)
        dy_dp = dyn.dy_dp(state, p)
        dy_dp = dy_dp[-3:, :]
        grad_output = grad_output.unsqueeze(0).t()
        grad_input = torch.tensor(dy_dp, requires_grad = True).t().mm(grad_output).t()

        return grad_input, None

Simulate = Simulate.apply

# Error for whole simulation
from numpy import linalg as LA

FE = 1e-6
p = torch.tensor(p, requires_grad = True).double()
y, p0 = Simulate(p)
#test = sum(y)
#test.backward()
y.backward(torch.FloatTensor([1.0, 1.0, 1.0]).double())
dy_dp = p.grad.double()
dy_dp_FD = np.zeros((1,len(dyn.p_init)))
#dy_dp_FD = np.zeros((3,len(dyn.p_init)))

for i in range(len(dyn.p_init)):
    dp= np.zeros(len(dyn.p_init))
    dp[i] = FE
    dp = torch.tensor(dp)
    y_p, pp = Simulate(p + dp)
    y_m, pm = Simulate(p - dp)
    y_p = y_p.detach().numpy()
    y_m = y_m.detach().numpy()
    dy_dp_FD[0, i] = sum((y_p - y_m) / (2* FE))
    #dy_dp_FD[:, i] = (y_p - y_m) / (2* FE)

dy_dp = dy_dp.detach().numpy()
err = LA.norm(dy_dp_FD - dy_dp)
print(f'The error is: {err:10.8f}')