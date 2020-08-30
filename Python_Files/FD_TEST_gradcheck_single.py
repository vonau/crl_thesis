import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
import pydde as d

#Parameters
samplenum = 1
input_size = 3
output_size = 3
time_length = 3 #seconds

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
        #print(f'input: {input.shape}')
        p = input.clone().numpy().transpose()
        state = dyn.compute(p)
        y_pred = torch.tensor(state.y[-3:], requires_grad = True)
        #print(f'y_pred: {y_pred.dtype}')
        
        ctx.save_for_backward(input)
        
        return y_pred
    
    @staticmethod
    def backward(ctx, grad_output):
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

#GRADCHECK
print('perfroming gradcheck...')
from torch.autograd import gradcheck

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
#p = dyn.p_init
p = torch.tensor(p, requires_grad = True)
input = (p.double())
#print(input)
test = gradcheck(Simulate, (input,), eps=1e-6, atol=1e-7, raise_exception = True)
print(test)