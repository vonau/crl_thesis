#######################################
# IMPORT LIBRARIES
import pydde as dde
import torch

#######################################
# LOAD SIMULATION AND OBJECTIVE FUNCTION
dyn = dde.DynamicSequence()
dyn.loadFile(simulation_file_path, nTimeSteps)

##########################################
#BUILD CUSTOM SIMULATION FUNCTION
class Simulate(torch.autograd.Function):
    def forward(ctx, p, data_input):
        dyn.q0 = data_input
        state = dyn.q(p)
        ctx.save_for_backward(p, data_input_)
        return state.q
        
    def backward(ctx, grad_output):
        p, data_input = ctx.saved_tensors
        dyn.q0 = data_input
        state = dyn.q(p)
        dq_dp = dyn.dq_dp(state, p)
        grad_input = grad_output.mm(dq_dp)
        return grad_input, None