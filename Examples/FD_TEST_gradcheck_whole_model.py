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
time_length = 60; #seconds

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
        
        return y_pred
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        p = input.clone().numpy().transpose()
        state= dyn.compute(p)
        dy_dp = dyn.dy_dp(state, p)
        dy_dp = dy_dp[-3:, :]
        grad_output = grad_output.unsqueeze(0).t()        
        grad_input = torch.tensor(dy_dp, requires_grad = True).t().mm(grad_output).t()

        return grad_input

Simulate = Simulate.apply
class ActiveLearn(nn.Module):

    def __init__(self, n_in, out_sz):
        super(ActiveLearn, self).__init__()

        self.L_in = nn.Linear(n_in, 3*time_length).double()
        self.Relu = nn.ReLU(inplace=True).double()
        self.P = nn.Linear(3*time_length, 3*time_length).double()
    
    def forward(self, input):
        x = self.L_in(input)
        x = self.Relu(x)
        x = self.P(x)
        x = self.Relu(x)
        x = Simulate(x)
        return x
    
model = ActiveLearn(input_size, output_size)

from torch.autograd import gradcheck

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
print('perfroming gradcheck...')
input = y_target.double()
test = gradcheck(model, (input,), eps=1e-6, atol=1e-7, raise_exception = True)
print(test)