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
epochs = 200
minibatch_size= 10
samplenum_target = 100
input_size = 15 # target (0:3), q (3:6), qdot (6:9), qddot (9:12), p_now (12:15) 
hiddenlayers = [180, 200]
learning_rate = 0.001
LRdecay = 0.7
use_case = 'varstart'
model_file_path = '../Trained_Models/'
sample_file_path = f'../Data/Samples/data_{use_case}_{nTimeSteps}tsteps_1507/'
simulation_file_path = '../Data/Simulations/pm_stiff.sim'
objective_file_path = f'../Data/Objectives/pm_target.obj'

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
for i in range(nTimeSteps):
	p_init[i*dyn.p0.size : (i+1)*dyn.p0.size] = dyn.p0
state_init = dyn.q(p_init)
r = dyn.r(state_init, p_init)
dr = dyn.dr_dp(state_init, p_init)
dq = dyn.dq_dp(state_init, p_init)

#########################################
#LOAD TRAINING SAMPLES
output_size = dyn.nParameters*nTimeSteps

number_of_files = len(os.listdir(sample_file_path))-4
samplenum = 1000*number_of_files
output_size = dyn.nParameters*nTimeSteps

input = np.zeros((samplenum, input_size))

for filenum in range(number_of_files):
    with open(sample_file_path + f'data_{filenum}.json') as json_file:
        data = json.load(json_file)
        filesize = len(data['q'])
        for i, q_i in enumerate(data['q_target']):
            input[filenum*filesize+i, 0:3] = np.array(q_i)
        for i, q_i in enumerate(data['q']):
            input[filenum*filesize+i, 3:6] = np.array(q_i)
        for i, qdot_i in enumerate(data['qdot']):
            input[filenum*filesize+i, 6:9] = np.array(qdot_i)
        for i, qddot_i in enumerate(data['qddot']):
            input[filenum*filesize+i, 9:12] = np.array(qddot_i)
        for i, p_now_i in enumerate(data['p_now']):
            input[filenum*filesize+i, 12:15] = np.array(p_now_i) - input[filenum*filesize+i, 3:6]

#Remove zeros
input = input[~(input == 0).all(1)]
print(f'Shape of dataafter removing faulty samples: {input.shape}')

data = input[0:samplenum_target, :]
samplenum = samplenum_target
print(f'Shape of data: {data.shape}')

##########################################
#BUILD CUSTOM SIMULATION FUNCTION
class Simulate(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, data_input):
        p = input.detach().clone().numpy()
        bs = len(p[:,0])
        q_pred = torch.ones([bs,dyn.nDofs*nTimeSteps])
        for i in range(bs):
            dyn.q0 = data_input[i, 3:6]
            dyn.qdot0 = data_input[i, 6:9]
            dyn.qddot0 = data_input[i, 9:12]
            dyn.p0 = data_input[i, 12:15] + data_input[i, 3:6]
            state = dyn.q(p[i,:])
            q_pred[i, :] = torch.tensor(state.q)
        data_input_ = torch.tensor(data_input)
        ctx.save_for_backward(input, data_input_)
        
        return q_pred
        
    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        input, data_input = ctx.saved_tensors
        p = input.detach().clone().numpy()
        bs = len(p[:,0])
        data_input = data_input.detach().clone().numpy()
        dq_dp_batch = torch.zeros([dyn.nDofs*nTimeSteps, dyn.nParameters*nTimeSteps])
        for i in range(bs):
            dyn.q0 = data_input[i, 3:6]
            dyn.qdot0 = data_input[i, 6:9]
            dyn.qddot0 = data_input[i, 9:12]
            dyn.p0 = data_input[i, 12:15] + data_input[i, 3:6]
            state = dyn.q(p[i, :])
            dq_dp = dyn.dq_dp(state, p[i, :])
            dq_dp = torch.tensor(dq_dp)
            dq_dp_batch = dq_dp_batch + dq_dp
        #print(f'dq/dp_batch: {dy_dp_batch/samplenum}')
        
        grad_input = grad_output.mm(dq_dp_batch.float()/bs)
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

criterion = nn.SmoothL1Loss(reduction = 'sum') 
#criterion = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size = 15, gamma=LRdecay, last_epoch=-1)
print("done")

####################################################
#TRAIN THE MODEL
torch.autograd.set_detect_anomaly(True)

start_time = time.time()
weight_c1 = 10 # q error
weight_c2 = 10 # p start condition
weight_c3 = 1 # p smoothness condition
batch = np.floor(samplenum/minibatch_size).astype(int)
losses= []
smoothness_errors_p = []
p_start_errors = []
q_errors = [] #q_end error
for e in range(epochs):
    for b in range(batch):
        input_numpy = data[b*minibatch_size:b*minibatch_size+minibatch_size,:]
        input_tensor = torch.tensor(input_numpy).float()
        p_b = model(input_tensor)
        q_pred = Simulate(p_b, input_numpy)
        #separate losses
        smoothness_error_p = weight_c3*criterion(p_b[:, 0:dyn.nParameters*(nTimeSteps-1)], p_b[:, dyn.nParameters:dyn.nParameters*nTimeSteps])
        p_start_error = weight_c2*criterion(p_b[:, 0:dyn.nParameters], input_tensor[:,12:15] + input_tensor[:,3:6])
        q_error = weight_c1*criterion(q_pred[:,-3:], input_tensor[:, 0:3])
        loss = q_error + p_start_error + smoothness_error_p
        losses.append(loss)
        smoothness_errors_p.append(smoothness_error_p)
        q_errors.append(q_error)
        p_start_errors.append(p_start_error)
        optimizer.zero_grad()
        #Backpropagation
        loss.backward()
        optimizer.step()
    if e%(epochs/10) == 0:
        print(f'epoch: {e:3}/{epochs}  loss: {loss.item():10.8f}   basic_loss: {q_error.item():10.8f}')

print(f'epoch: {e:3} finale loss: {loss.item():10.8f}') # print the last line
print(f'\nDuration: {(time.time() - start_time)/60:.3f} min') # print the time elapsed


##################################################
#PLOT EVERY LOSS COMPONENT FOR EACH BATCH
timestr = time.strftime("%H%M")
# epoch_lines = np.arange(0, epochs, 1/batch)
epoch_i = np.arange(epochs*batch)/batch
plt.figure(figsize = [12,6])
loss = plt.plot(epoch_i, losses, label = 'total loss', linewidth=3)
q_end = plt.plot(epoch_i, q_errors, label = 'target error', linestyle='--')
smoothness = plt.plot(epoch_i, smoothness_errors_p, label = 'smoothness error', linestyle='--')
p_start = plt.plot(epoch_i, p_start_errors, label = 'p_start error', linestyle='--')
plt.legend()
plt.yscale('log')
plt.ylabel('Loss (Summed Squared Error)')
plt.xlabel('Epochs')
# for xc in epoch_lines:
#     plt.axvline(x=xc, linewidth = 0.2)
plt.savefig('../Plots/Loss_' + use_case + f'_{samplenum_target}_' + timestr + '.png')

#####################################################
#SAVE MODEL
timestr = time.strftime("%m%d")
#Save entire Model
if samplenum_target == 15000:
    torch.save(model, model_file_path + 'Model_active_' + use_case + f'_{nTimeSteps}tsteps_{samplenum}s_{minibatch_size}bs_{epochs}e_{LRdecay}lr_' + timestr + '.pt')
    torch.save(model, model_file_path + 'Model_active_' + use_case + f'_{nTimeSteps}tsteps_latest.pt')

    #Save parameters of Model
    torch.save(model.state_dict(), model_file_path + 'state_dict/Model_statedict_active_' + use_case + f'_{nTimeSteps}tsteps_{samplenum}s_{minibatch_size}bs_{epochs}e_{LRdecay}lr_' + timestr + '.pt')
    torch.save(model.state_dict(), model_file_path + 'state_dict/Model_statedict_active_' + use_case + f'_{nTimeSteps}tsteps_latest.pt')

    #Convert to Torch Script and save for CPP application
    input_example = input_tensor[4, :]
    traced_script_module = torch.jit.trace(model, input_example)

    # Test the torch script
    #test_input = torch.tensor([0, 2, 0.5])
    #original = model(test_input)
    #output_example = traced_script_module(test_input)

    traced_script_module.save(model_file_path + 'Serialized_Models/Serialized_model_active_' + use_case + f'_{nTimeSteps}tsteps_latest.pt')
    traced_script_module.save(model_file_path + 'Serialized_Models/Serialized_model_active_' + use_case + f'_{nTimeSteps}tsteps_{samplenum}s_{minibatch_size}bs_{epochs}e_{LRdecay}lr_' + timestr + '.pt')
    print('Model saved')

else:
    #Save entire Model
    torch.save(model, model_file_path + 'Model_active_' + use_case + f'_{nTimeSteps}tsteps_{samplenum}s_{minibatch_size}bs_{epochs}e_{LRdecay}lr_' + timestr + f'_{samplenum_target}.pt')
    torch.save(model, model_file_path + 'Model_active_' + use_case + f'_{nTimeSteps}tsteps_latest_{samplenum_target}.pt')

    #Save parameters of Model
    torch.save(model.state_dict(), model_file_path + 'state_dict/Model_statedict_active_' + use_case + f'_{nTimeSteps}tsteps_{samplenum}s_{minibatch_size}bs_{epochs}e_{LRdecay}lr_' + timestr + f'_{samplenum_target}.pt')
    torch.save(model.state_dict(), model_file_path + 'state_dict/Model_statedict_active_' + use_case + f'_{nTimeSteps}tsteps_latest_{samplenum_target}.pt')

    #Convert to Torch Script and save for CPP application
    input_example = input_tensor[4, :]
    traced_script_module = torch.jit.trace(model, input_example)

    # Test the torch script
    #test_input = torch.tensor([0, 2, 0.5])
    #original = model(test_input)
    #output_example = traced_script_module(test_input)

    traced_script_module.save(model_file_path + 'Serialized_Models/Serialized_model_active_' + use_case + f'_{nTimeSteps}tsteps_latest_{samplenum_target}.pt')
    traced_script_module.save(model_file_path + 'Serialized_Models/Serialized_model_active_' + use_case + f'_{nTimeSteps}tsteps_{samplenum}s_{minibatch_size}bs_{epochs}e_{LRdecay}lr_' + timestr + f'_{samplenum_target}.pt')
    print('Model saved')