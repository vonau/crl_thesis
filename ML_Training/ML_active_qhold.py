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
print(time.strftime("%H%M"))
nTimeSteps = 60 #seconds
print(f'nTimeSteps: {nTimeSteps}')
samplenum = 15000
samplenum_target = 100
epochs = 30
minibatch_size= 10
input_size = 9
hiddenlayers = [180, 200]
learning_rate = 0.001
LRdecay = 0.7
use_case = 'qhold'
model_file_path = '../Trained_Models/'
sample_file_path = f'../Data/Samples/data_{use_case}_{nTimeSteps}tsteps_1024/'
simulation_file_path = '../Data/Simulations/pm_target.sim'
objective_file_path = f'../Data/Objectives/pm_qhold.obj'

# check dde version
print("using dde version: " + dde.__version__)
# set log level
dde.set_log_level(dde.LogLevel.off)
print(f'log level set to {dde.get_log_level()}')

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

# Objective Function
obj = dde.InverseObjective(dyn)
obj.loadFile(objective_file_path)
objective_json = json.load(open(objective_file_path))
opt = dde.Newton()

#########################################
#LOAD TRAINING SAMPLES
number_of_files = len(os.listdir(sample_file_path))-4
samplenum = 1000*number_of_files
output_size = dyn.nParameters*nTimeSteps

p = np.zeros((samplenum, dyn.nParameters*nTimeSteps))
input = np.zeros((samplenum, input_size))

for filenum in range(number_of_files):
    with open(sample_file_path + f'data_{filenum}.json') as json_file:
        data_ = json.load(json_file)
        filesize = len(data_['q'])
        for i, p_i in enumerate(data_['p']):
            p[filenum*filesize+i, :] = np.array(p_i)
        for i, q_i in enumerate(data_['q']):
            input[filenum*filesize+i, 0:3] = np.array(q_i)
        for i, qdot_i in enumerate(data_['qdot']):
            input[filenum*filesize+i, 3:6] = np.array(qdot_i)
        for i, p_now_i in enumerate(data_['p_now']):
            input[filenum*filesize+i, 6:9] = np.array(p_now_i)-input[filenum*filesize+i, 0:3]

print(f'Shape of input: {input.shape}')
print(f'Shape of p: {p.shape}')
#Remove zeros
data = input[~(input == 0).all(1)]
p = p[~(p == 0).all(1)]
data = data[0:samplenum_target, :]
samplenum = samplenum_target
print(data.shape)
print(p.shape)

p = torch.tensor(p).float()

##########################################
#BUILD CUSTOM SIMULATION FUNCTION
class Simulate(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, data_input):
        #print(f'input: {input.shape}')
        p = input.detach().clone().numpy()
        q_pred = torch.ones([len(p[:, 0]),dyn.nParameters*nTimeSteps])
        for i in range(len(p[:, 0])):
            dyn.q0 = data_input[i, 0:3]
            dyn.qdot0 = data_input[i, 3:6]
            dyn.po = data_input[i, 6:9]
            state = dyn.q(p[i, :])
            q_pred[i, :] = torch.tensor(state.q)
        #print(f'q_pred: {q_pred.shape}')
        data_input_ = torch.tensor(data_input)
        ctx.save_for_backward(input, data_input_)
        
        return q_pred
        
    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        input, data_input = ctx.saved_tensors
        p = input.detach().clone().numpy()
        data_input = data_input.detach().clone().numpy()
        dq_dp_batch = torch.zeros([dyn.nDofs*nTimeSteps, dyn.nParameters*nTimeSteps])
        for i in range(len(p[:, 0])):
            dyn.q0 = data_input[i, 0:3]
            dyn.qdot0 = data_input[i, 3:6]
            dyn.po = data_input[i, 6:9]
            state = dyn.q(p[i, :])
            dq_dp = dyn.dq_dp(state, p[i, :])
            dq_dp = torch.tensor(dq_dp)
            dq_dp_batch = dq_dp_batch + dq_dp
        #print(f'dq/dp_batch: {dy_dp_batch/samplenum}')
        grad_input = grad_output.mm(dq_dp_batch.float()/len(p[:,0]))
        #print(f'shape of grad input: {grad_input.shape}')
        #print(f'shape of grad output: {grad_output.shape}')
        return grad_input, None

Simulate = Simulate.apply


########################################
#BUILD CUSTOM MODEL
class ActiveLearn(nn.Module):

    def __init__(self, n_in, out_sz):
        super(ActiveLearn, self).__init__()

        self.L_in = nn.Linear(n_in, hiddenlayers[0])
        self.H1 = nn.Linear(hiddenlayers[0], hiddenlayers[1])
        self.H2 = nn.Linear(hiddenlayers[1], out_sz)
        #self.H3 = nn.Linear(h[2], 3*time_length)
        self.L_out = nn.Linear(out_sz, out_sz)
        self.Relu = nn.ReLU(inplace=True)
        #self.drop = nn.Dropout(p=0.3)
        #self.norm1 = nn.BatchNorm2d(h[0])
        #self.norm2 = nn.BatchNorm2d(h[1])
    
    def forward(self, input):
        x = self.L_in(input)
        #x = self.norm1(x)
        #x = self.drop(x)
        x = self.Relu(x)
        x = self.H1(x)
        #x = self.norm2(x)
        x = self.Relu(x)
        x = self.H2(x)
        x = self.Relu(x)
        #x = self.H3(x)
        #x = self.Relu(x)
        x = self.L_out(x)
        return x


model = ActiveLearn(input_size, output_size)

#criterion = nn.SmoothL1Loss()  # RMSE = np.sqrt(MSE)
criterion = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size = 15, gamma=LRdecay, last_epoch=-1)
print("done")

####################################################
#TRAIN THE MODEL
torch.autograd.set_detect_anomaly(True)

start_time = time.time()
weight_c1 = 1 # q error
weight_c2 = 1 # p start condition
weight_c3 = 1 # p smoothness condition
weight_c4 = 100 # p smoothness condition
batch = np.floor(samplenum/minibatch_size).astype(int)
losses= []
smoothness_errors_p = []
smoothness_errors_q = []
p_start_errors = []
q_errors = [] #q_end error
for e in range(epochs):
    for b in range(batch):
        input_numpy = data[b*minibatch_size:b*minibatch_size+minibatch_size,:]
        input_tensor = torch.tensor(data[b*minibatch_size:b*minibatch_size+minibatch_size,:], requires_grad = True).float()
        p_b = model(input_tensor)
        q_pred = Simulate(p_b, input_numpy)
        #smoothness_error_i = weight_c3*(p_i[0:3*(nTimeSteps-1)] - p_i[3:3*nTimeSteps]).pow(2).sum()
        smoothness_error_p = weight_c3*criterion(p_b[:, 0:dyn.nParameters*(nTimeSteps-1)], p_b[:, dyn.nParameters:dyn.nParameters*nTimeSteps])
        smoothness_error_q = weight_c4*criterion(q_pred[:, 0:dyn.nDofs*(nTimeSteps-1)], q_pred[:, dyn.nDofs:dyn.nDofs*nTimeSteps])
        #p_start_error = weight_c2*torch.sqrt(criterion(p_i[0:3], torch.tensor(dyn.p_init[0:3])))
        p_start_error = weight_c2*criterion(p_b[:, 0:dyn.nParameters], input_tensor[:,6:9]+input_tensor[:,0:3])
        #q_end_error = torch.sqrt(criterion(q_pred, q_i))
        #q_error = weight_c1*criterion(q_pred, q_truth[b*minibatch_size:b*minibatch_size+minibatch_size,:])
        #loss = q_error + p_start_error + smoothness_error_p + smoothness_error_q
        loss = p_start_error + smoothness_error_p + smoothness_error_q
        #loss = loss_batch/minibatch_size
        losses.append(loss)
        #smoothness_error = smoothness_error_batch/minibatch_size
        smoothness_errors_p.append(smoothness_error_p)
        smoothness_errors_q.append(smoothness_error_q)
        #q_errors.append(q_error)
        p_start_errors.append(p_start_error)
        optimizer.zero_grad()
        #Backpropagation
        loss.backward()
        optimizer.step()
    if e%(epochs/5) == 0:
        print(f'epoch: {e:3}/{epochs}  loss: {loss.item():10.8f}   basic_loss: {smoothness_error_q.item():10.8f}')

print(f'epoch: {e:3} finale loss: {loss.item():10.8f}') # print the last line
print(f'\nDuration: {(time.time() - start_time)/60:.3f} min') # print the time elapsed

##################################################
#PLOT EVERY LOSS COMPONENT FOR EACH BATCH
timestr = time.strftime("%H%M")
# epoch_lines = np.arange(0, epochs, 1/batch)
epoch_i = np.arange(epochs*batch)/batch
plt.figure(figsize = [12,6])
loss = plt.plot(epoch_i, losses, label = 'total loss', linewidth=3)
smoothness = plt.plot(epoch_i, smoothness_errors_q,  label = 'smoothness error', linestyle='--')
#q_end = plt.plot(epoch_i, q_errors, label = 'y error', linestyle='--')
p_start = plt.plot(epoch_i, p_start_errors, label = 'p_start error', linestyle='--')
plt.legend()
plt.yscale('log')
plt.ylabel('error')
plt.xlabel('epochs')
# for xc in epoch_lines:
#     plt.axvline(x=xc, linewidth = 0.2)
plt.savefig('../Plots/Loss_' + use_case + f'_{samplenum_target}_' + timestr + '.png')

#####################################################
#SAVE MODEL
timestr = time.strftime("%m%d")
#Save entire Model
torch.save(model, model_file_path + 'Model_active_' + use_case + f'_{nTimeSteps}tsteps_{samplenum}s_{epochs}e_{LRdecay}lr_' + timestr + f'_{samplenum_target}.pt')
torch.save(model, model_file_path + 'Model_active_' + use_case + f'_{nTimeSteps}tsteps_latest_{samplenum_target}.pt')

#Save parameters of Model
torch.save(model.state_dict(), model_file_path + 'state_dict/Model_statedict_active_' + use_case + f'_{nTimeSteps}tsteps_{samplenum}s_{epochs}e_{LRdecay}lr_' + timestr + f'_{samplenum_target}.pt')
torch.save(model.state_dict(), model_file_path + 'state_dict/Model_statedict_active_' + use_case + f'_{nTimeSteps}tsteps_latest_{samplenum_target}.pt')

#Convert to Torch Script and save for CPP application
input_example = input_tensor[4, :]
traced_script_module = torch.jit.trace(model, input_example)

# Test the torch script
#test_input = torch.tensor([0, 2, 0.5])
#original = model(test_input)
#output_example = traced_script_module(test_input)

traced_script_module.save(model_file_path + 'Serialized_Models/Serialized_model_active_' + use_case + f'_{nTimeSteps}tsteps_latest_{samplenum_target}.pt')
traced_script_module.save(model_file_path + 'Serialized_Models/Serialized_model_active_' + use_case + f'_{nTimeSteps}tsteps_{samplenum}s_{epochs}e_{LRdecay}lr_' + timestr + f'_{samplenum_target}.pt')
print('Model saved')