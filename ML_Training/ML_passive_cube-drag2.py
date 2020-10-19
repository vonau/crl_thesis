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
nTimeSteps = 60 #at 60 Hz
epochs = 300
minibatch_size= 20
samplenum = 10000
samplenum_test = 5000
hiddenlayers = [280, 350]
samples_per_file = 1000
input_size = 25 # 0:3 target, 3:9 q, 9:15 qdot, 15:21 qddot, 21:25 p_now
output_size = 4*nTimeSteps
use_case = 'cube-drag'
learning_rate = 0.001
LRdecay = 0.9
model_file_path = '../Trained_Models/'
model_file_path_passive = '../Trained_Models/state_dict/Model_statedict_passive_' + use_case + f'_{nTimeSteps}tsteps_latest.pt'
sample_file_path = f'../Data/Samples/data_cube-drag_{nTimeSteps}tsteps_2315/'
simulation_file_path = '../Data/Simulations/cube-drag.sim'
objective_file_path = f'../Data/Objectives/cube-drag.obj'
# set log level
dde.set_log_level(dde.LogLevel.off)
print(os.listdir(sample_file_path))

############################################
#LOAD SIMULATION PYDDE_V2
dyn = dde.DynamicSequence()
dyn.loadFile(simulation_file_path, nTimeSteps)
p_init_0 = np.zeros(dyn.p0.size*nTimeSteps)
for i in range(0,nTimeSteps):
	p_init_0[i*dyn.p0.size : (i+1)*dyn.p0.size] = dyn.p0
state_init = dyn.q(p_init_0)
r = dyn.r(state_init, p_init_0)
dr = dyn.dr_dp(state_init, p_init_0)
dq = dyn.dq_dp(state_init, p_init_0)

#########################################
#LOAD TRAINING SAMPLES WITHOUT QDDOT
# 0:3 target, 3:9 q, 9:15 qdot, 15:19 p_now
number_of_files = len(os.listdir(sample_file_path))-4
samplenum_temp = samples_per_file*number_of_files

p = np.zeros((samplenum_temp, dyn.nParameters*nTimeSteps))
input = np.zeros((samplenum_temp, input_size))

for filenum in range(number_of_files):
    with open(sample_file_path + f'data_{filenum}.json') as json_file:
        data = json.load(json_file)
        filesize = len(data['q_target'])
        for i, p_i in enumerate(data['p']):
            p[filenum*filesize+i, :] = np.array(p_i)
        for i, q_target_i in enumerate(data['q_target']):
            input[filenum*filesize+i, 0:3] = np.array(q_target_i)
        for i, q_i in enumerate(data['q']):
            input[filenum*filesize+i, 3:9] = np.array(q_i)
        for i, qdot_i in enumerate(data['qdot']):
            input[filenum*filesize+i, 9:15] = np.array(qdot_i)
        for i, qddot_i in enumerate(data['qddot']):
            input[filenum*filesize+i, 15:21] = np.array(qddot_i)
        for i, p_now_i in enumerate(data['p_now']):
            input[filenum*filesize+i, 21:25] = np.array(p_now_i) - [input[filenum*filesize+i, 3], input[filenum*filesize+i, 5], input[filenum*filesize+i, 6], input[filenum*filesize+i, 8]]

print(f'Shape of input: {input.shape}')
print(f'Shape of p: {p.shape}')
#Remove zeros
p_data = p[~(p == 0).all(1)]
data = input[~(input == 0).all(1)]
p_data = p_data[0:samplenum, :]
data = data[0:samplenum, :]
print(data.shape)
print(p_data.shape)
'''
#normalize qddot
def minmaxscale(input, extrema):
    if extrema == None:
        maximas= []
        minimas= []
        for i in range(len(input[0, :])):
            maximas.append(np.max(input[:,i]))
            minimas.append(np.min(input[:,i]))
        max = np.max(maximas)
        min = np.min(minimas)
        extrema = np.max([max, np.linalg.norm(min)])
        scaled = (input+extrema)/(2*extrema)
        return scaled, extrema
    else:
        scaled = (input+extrema)/(2*extrema)
        return scaled

data[:, 15:18], extr_qddot = minmaxscale(data[:, 15:18], None)
data[:, 18:21], extr_qddot = minmaxscale(data[:, 18:21], None)
print("15:18")
print(data[:, 15:18])
'''
'''
maximas_qddot_1 = [np.max(input[:, 15]), np.max(input[:, 16]), np.max(input[:, 17]), np.max(input[:, 18], np.max(input[:, 19], np.max(input[:, 20]]
minimas_qddot_1 = [np.min(input[:, 15]), np.min(input[:, 16]), np.min(input[:, 17]), np.min(input[:, 18], np.min(input[:, 19], np.min(input[:, 20]]
max_scale_qddot = np.max(maximas_qddot)
minscale_qddot = np.min(minimas_qddot)
extrema_qddot = np.max([max_scale_qddot, np.linalg.norm(minscale_qddot)])
print(max_scale_qddot)
print(minscale_qddot)
print(extrema_qddot)
input[:, 9:12] = (input[:, 15:21]+extrema_qddot)/(2*extrema_qddot)
'''
#print("18:21")
#print(data[:, 18:21])
# Splitting the dataset into the Training set and Test set

input = torch.tensor(input).float()
p = torch.tensor(p).float()

#########################################
#LOAD TEST SAMPLES
# 0:3 target, 3:9 q, 9:15 qdot, 15:19 p_now
number_of_files_test = len(os.listdir(sample_file_path + 'data_test/'))
samplenum_test_temp = 1000*number_of_files_test

p_test = np.zeros((samplenum_test_temp, dyn.nParameters*nTimeSteps))
input_test = np.zeros((samplenum_test_temp, input_size))

for filenum in range(number_of_files_test):
    with open(sample_file_path + f'data_test/data_{filenum}.json') as json_file:
        data = json.load(json_file)
        filesize = len(data['q_target'])
        for i, p_i in enumerate(data['p']):
            p_test[filenum*filesize+i, :] = np.array(p_i)
        for i, q_target_i in enumerate(data['q_target']):
            input_test[filenum*filesize+i, 0:3] = np.array(q_target_i)
        for i, q_i in enumerate(data['q']):
            input_test[filenum*filesize+i, 3:9] = np.array(q_i)
        for i, qdot_i in enumerate(data['qdot']):
            input_test[filenum*filesize+i, 9:15] = np.array(qdot_i)
        for i, qddot_i in enumerate(data['qddot']):
           input_test[filenum*filesize+i, 15:21] = np.array(qddot_i)
        for i, p_now_i in enumerate(data['p_now']):
            input_test[filenum*filesize+i, 21:25] = np.array(p_now_i) - [input_test[filenum*filesize+i, 3],input_test[filenum*filesize+i, 5], input_test[filenum*filesize+i, 6], input_test[filenum*filesize+i, 8]]

print(f'Shape of input_test: {input_test.shape}')
print(f'Shape of p_test: {p_test.shape}')
#Remove zeros
p_test = p_test[~(input_test == 0).all(1)]
input_test = input_test[~(input_test == 0).all(1)]
print(f'Shape of DATA_TEST after removing failed samples:{input_test.shape}')
print(f'Shape of P_TEST after removing failed samples:{p_test.shape}')
input_test = input_test[0:samplenum_test, :]
p_test = p_test[0:samplenum_test, :]

p_test = torch.tensor(p_test).float()

print(f'Shape of input: {input_test.shape}')
print(f'Shape of p: {p_test.shape}')

##########################################
#BUILD CUSTOM SIMULATION FUNCTION
class Simulate(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, data_input):
        p_ = input.detach().clone().numpy()
        bs = len(p_[:,0])
        q_pred = torch.ones([len(p_[:, 0]),dyn.nDofs*nTimeSteps])
        for i_f in range(bs):
            dyn.q0 = data_input[i_f, 3:9]
            dyn.qdot0 = data_input[i_f, 9:15]
            dyn.qddot0 = data_input[i_f, 15:21]
            state = dyn.q(p_[i_f,:])
            q_pred[i_f, :] = torch.tensor(state.q)
        data_input_ = torch.tensor(data_input)
        ctx.save_for_backward(input, data_input_)
        
        return q_pred
        
    @staticmethod
    def backward(ctx, grad_output):
        input, data_input = ctx.saved_tensors
        p_2 = input.detach().clone().numpy()
        bs_ = len(p_2[:,0])
        data_input_2 = data_input.detach().clone().numpy()
        dq_dp_batch = torch.zeros([dyn.nDofs*nTimeSteps, dyn.nParameters*nTimeSteps])
        for i_b in range(bs_):
            dyn.q0 = data_input_2[i_b, 3:9]
            dyn.qdot0 = data_input_2[i_b, 9:15]
            dyn.qddot0 = data_input_2[i_b, 15:21]
            state = dyn.q(p_2[i_b, :])
            dq_dp = dyn.dq_dp(state, p_2[i_b, :])
            dq_dp = torch.tensor(dq_dp)
            dq_dp_batch = dq_dp_batch + dq_dp
        test = dq_dp_batch.float()/bs_
        grad_input = grad_output.mm(test)
        return grad_input, None

Simulate = Simulate.apply

################################
#BUILD MODEL
class PassiveLearn(nn.Module):

    def __init__(self, n_in, out_sz):
        super(PassiveLearn, self).__init__()

        self.L_in = nn.Linear(n_in, hiddenlayers[0])
        self.H1 = nn.Linear(hiddenlayers[0], hiddenlayers[1])
        self.H2 = nn.Linear(hiddenlayers[1], out_sz)
        self.L_out = nn.Linear(out_sz, out_sz)
        self.Relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.5)
    
    def forward(self, input):
        x = self.L_in(input)
        x = self.Relu(x)
        x = self.H1(x)
        x = self.Relu(x)
        x = self.H2(x)
        x = self.Relu(x)
        x = self.L_out(x)
        return x

model = PassiveLearn(input_size, output_size)

#criterion = nn.SmoothL1Loss()  # RMSE = np.sqrt(MSE)
criterion = nn.MSELoss(reduction= 'sum')  # RMSE = np.sqrt(MSE)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size = 15, gamma=LRdecay, last_epoch=-1)

################################################
#TRAIN THE MODEL

torch.autograd.set_detect_anomaly(True)

start_time = time.time()
weight_c1 = 1 # p error condition
batch = np.floor(samplenum/minibatch_size).astype(int)
losses= []
p_smoothness_errors = []
basic_errors = [] #y_end_ and p_start error
for e in range(epochs):
    for b in range(batch):
        loss_batch = 0
        input_i = input[b*minibatch_size:b*minibatch_size+minibatch_size,:]
        p_i = p[b*minibatch_size:b*minibatch_size+minibatch_size,:]
        p_pred = model(input_i)
        #error calculation
        loss_batch = weight_c1 * criterion(p_pred, p_i)
        losses.append(loss_batch)
        optimizer.zero_grad()
        #Back Prop
        loss_batch.backward()
        optimizer.step()
    scheduler.step()
    LR= scheduler.get_lr()
    #print(loss_batch)
    if e%(epochs/5) == 0:
        print(f'epoch: {e:3}/{epochs}    LR: {LR[0]:10.6f}  loss: {loss_batch:10.8f}')

print(f'epoch: {e:3} final loss: {loss_batch.item():10.8f}') # print the last line
print(f'\nTraining completed. Total duration: {(time.time() - start_time)/60:.3f} min') # print the time elapsed

##################################################
#Plot loss for each batch after each epoch
timestr = time.strftime("%H%M")

epoch_i = np.arange(epochs*batch)/batch
plt.figure(figsize = [12,6])
loss = plt.plot(losses, label = 'total loss', linewidth=3)
plt.legend()
plt.yscale('log')
plt.ylabel('Loss (Summed Squared Error)')
plt.xlabel('Epochs')
# for xc in epoch_lines:
#     plt.axvline(x=xc, linewidth = 0.2)
plt.savefig('Loss_passive_training_' + use_case + f'_{samplenum}_' + timestr + '.png')


####################################################
#Test the data
input_testTensor = torch.tensor(input_test).float()
samplenum_test = len(input_test[:,0])
losses_test= []
with torch.no_grad():
    for i in range(samplenum_test):
        p_val = model(input_testTensor[i, :])
        loss_test = criterion(p_val,p_test[i,:])
        losses_test.append(loss_test.clone().numpy())
#plot test errors
plt.figure(figsize = [12,6])
loss_test_plot = plt.plot(losses_test, label = 'loss')
plt.legend()
plt.ylabel('loss')
plt.xlabel('samples')
tot_error = sum(losses_test)
mean_error = np.mean(losses_test)
print(f'TOTAL ERROR TO SAMPLES: {tot_error}    MEAN ERROR TO SAMPLES: {mean_error}') 

criterion2 = torch.nn.MSELoss(reduction = 'sum')
losses_test= []
with torch.no_grad():
    for i in range(samplenum_test):
        p_val = model(input_testTensor[i, :])
        p_val_ = torch.unsqueeze(p_val, 0)
        q_val = Simulate(p_val_, np.expand_dims(input_test[i, :], axis=0))
        pmq = torch.tensor([input[i, 3], input[i, 5], input[i, 6], input[i, 8]])
        p_start_error = criterion2(p_val[0:dyn.nParameters], input_testTensor[i,21:25] + pmq)
        q_error = criterion2(q_val[0, dyn.nDofs*(nTimeSteps-1):dyn.nDofs*(nTimeSteps-1)+3], input_testTensor[i, 0:3])
        smoothness_error = criterion2(p_val[0:dyn.nParameters*(nTimeSteps-1)], p_val[dyn.nParameters:dyn.nParameters*nTimeSteps])
        loss2 = p_start_error + q_error + smoothness_error
        losses_test.append(loss2.clone().numpy())
tot_error = sum(losses_test)
mean_error = np.mean(losses_test)
print(f'TOTAL ERROR: {tot_error}    MEAN ERROR: {mean_error}    Batchsize: {minibatch_size}') 

#####################################################
#SAVE MODEL
samplenum_target = samplenum
if samplenum_target == 15000:
    timestr = time.strftime("%m%d")
    #Save entire Model
    torch.save(model, model_file_path + 'Model_passive_' + use_case + f'_{nTimeSteps}tsteps_{samplenum}s_{epochs}e_{LRdecay}lr_' + timestr + '.pt')
    torch.save(model, model_file_path + 'Model_passive_' + use_case + f'_{nTimeSteps}tsteps_latest.pt')

    #Save parameters of Model
    torch.save(model.state_dict(), model_file_path + 'state_dict/Trained_Model_statedict_passive_' + use_case + f'_{nTimeSteps}tsteps_{samplenum}s_{epochs}e_{LRdecay}lr_' + timestr + '.pt')
    torch.save(model.state_dict(), model_file_path + 'state_dict/Model_statedict_passive_' + use_case + f'_{nTimeSteps}tsteps_latest.pt')

    #Convert to Torch Script and save for CPP application
    input_example = input[4, :]
    traced_script_module = torch.jit.trace(model, input_example)

    # Test the torch script
    #test_input = torch.tensor([0, 2, 0.5])
    #original = model(test_input)
    #output_example = traced_script_module(test_input)

    traced_script_module.save(model_file_path + 'Serialized_Models/Serialized_model_passive_' + use_case + f'_{nTimeSteps}tsteps_latest.pt')
    traced_script_module.save(model_file_path + 'Serialized_Models/Serialized_model_passive_' + use_case + f'_{nTimeSteps}tsteps_{samplenum}s_{epochs}e_{LRdecay}lr_' + timestr + '.pt')
    print('Model saved')

else:
    timestr = time.strftime("%m%d")
    #Save entire Model
    torch.save(model, model_file_path + 'Model_passive_' + use_case + f'_{nTimeSteps}tsteps_{samplenum}s_{epochs}e_{LRdecay}lr_' + timestr + f'_{samplenum_target}.pt')
    torch.save(model, model_file_path + 'Model_passive_' + use_case + f'_{nTimeSteps}tsteps_latest_{samplenum_target}.pt')

    #Save parameters of Model
    torch.save(model.state_dict(), model_file_path + 'state_dict/Model_statedict_passive_' + use_case + f'_{nTimeSteps}tsteps_{samplenum}s_{epochs}e_{LRdecay}lr_' + timestr + f'_{samplenum_target}.pt')
    torch.save(model.state_dict(), model_file_path + 'state_dict/Model_statedict_passive_' + use_case + f'_{nTimeSteps}tsteps_latest_{samplenum_target}.pt')

    #Convert to Torch Script and save for CPP application
    input_example = input[4, :]
    traced_script_module = torch.jit.trace(model, input_example)

    # Test the torch script
    #test_input = torch.tensor([0, 2, 0.5])
    #original = model(test_input)
    #output_example = traced_script_module(test_input)

    traced_script_module.save(model_file_path + 'Serialized_Models/Serialized_model_passive_' + use_case + f'_{nTimeSteps}tsteps_latest_{samplenum_target}.pt')
    traced_script_module.save(model_file_path + 'Serialized_Models/Serialized_model_passive_' + use_case + f'_{nTimeSteps}tsteps_{samplenum}s_{epochs}e_{LRdecay}lr_' + timestr + f'_{samplenum_target}.pt')
    print('Model saved')