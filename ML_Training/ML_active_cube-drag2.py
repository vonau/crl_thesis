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
epochs = 2
minibatch_size= 1
samplenum_target = 10000
learning_rate = 0.000001
LRdecay = 0.5
use_case = 'cube-drag'
path = '/home/nico/Desktop/'
model_file_path = '../Trained_Models/'
model_file_path_passive = '../Trained_Models/state_dict/Model_statedict_passive_' + use_case + f'_{nTimeSteps}tsteps_latest.pt'
sample_file_path = path + f'Data/Samples/data_cube-drag_{nTimeSteps}tsteps_2315/'
simulation_file_path = path + 'Data/Simulations/cube-drag.sim'
objective_file_path = path + 'Data/Objectives/cube-drag.obj'

# check dde version
print("using dde version: " + dde.__version__)
# set log level
dde.set_log_level(dde.LogLevel.off)
print(f'log level set to {dde.get_log_level()}')

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

# Objective Function
obj = dde.InverseObjective(dyn)
obj.loadFile(objective_file_path)
objective_json = json.load(open(objective_file_path))
opt = dde.Newton()

##########################################
#SAMPLE TARGETS
input_size = 25 # target (0:3), q (3:9), qdot (9:15), qddot (15:21), p_now (21:25) 
output_size = dyn.nParameters*nTimeSteps
print(os.listdir(sample_file_path))

#########################################
#LOAD TRAINING SAMPLES
number_of_files = len(os.listdir(sample_file_path))-4

with open(sample_file_path + f'data_0.json') as json_file:
    data = json.load(json_file)
    filesize = len(data['q_target'])
samplenum = filesize*number_of_files
input = np.zeros((samplenum, input_size))
for filenum in range(number_of_files):
    with open(sample_file_path + f'data_{filenum}.json') as json_file:
        data = json.load(json_file)
        for i, q_target_i in enumerate(data['q_target']):
            input[filenum*filesize+i, 0:3] = np.array(q_target_i)
        for i, q_i in enumerate(data['q']):
            input[filenum*filesize+i, 3:9] = np.array(q_i)
            pmq = np.array([q_i[0],q_i[2], q_i[3], q_i[5]])
        for i, qdot_i in enumerate(data['qdot']):
            input[filenum*filesize+i, 9:15] = np.array(qdot_i)
        for i, qddot_i in enumerate(data['qddot']):
            input[filenum*filesize+i, 15:21] = np.array(qddot_i)
        for i, p_now_i in enumerate(data['p_now']):
            input[filenum*filesize+i, 21:25] = np.array(p_now_i) - pmq

print(f'Shape of input: {input.shape}')
#Remove zeros
data = input[~(input == 0).all(1)]
print(f'after removing zeros: {data.shape}')
data = data[0:samplenum_target, :]
samplenum = len(data[:,0])
print(data.shape)

##########################################
#BUILD CUSTOM SIMULATION FUNCTION
class Simulate(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, data_input):
        p_ = input.detach().clone().numpy()
        bs = len(p_[:,0])
        q_pred = torch.ones([bs,dyn.nDofs*nTimeSteps])
        for i_f in range(bs):
            dyn.q0 = data_input[i_f, 3:9]
            dyn.qdot0 = data_input[i_f, 9:15]
            dyn.qddot0 = data_input[i_f, 15:21]
            dyn.p0 = data_input[i_f, 21:25] + [data_input[i_f, 3],data_input[i_f, 5], data_input[i_f, 6], data_input[i_f, 8]]
            state = dyn.q(p_[i_f,:])
            q_pred[i_f, :] = torch.tensor(state.q)
        data_input_ = torch.tensor(data_input)
        ctx.save_for_backward(input, data_input_)
        
        return q_pred
        
    @staticmethod
    def backward(ctx, grad_output):
        input, data_input = ctx.saved_tensors
        p_2 = input.detach().clone().numpy()
        bs = len(p_2[:,0])
        data_input_2 = data_input.detach().clone().numpy()
        dq_dp_batch = torch.zeros([dyn.nDofs*nTimeSteps, dyn.nParameters*nTimeSteps])
        for i_b in range(bs):
            dyn.q0 = data_input_2[i_b, 3:9]
            dyn.qdot0 = data_input_2[i_b, 9:15]
            dyn.qddot0 = data_input_2[i_b, 15:21]
            dyn.p0 = data_input_2[i_b, 21:25] + [data_input_2[i_b, 3],data_input_2[i_b, 5], data_input_2[i_b, 6], data_input_2[i_b, 8]]
            state = dyn.q(p_2[i_b, :])
            dq_dp = dyn.dq_dp(state, p_2[i_b, :])
            dq_dp = torch.tensor(dq_dp)
            dq_dp_batch = dq_dp_batch + dq_dp
        test = dq_dp_batch.float()/bs
        # print(test)
        grad_input = grad_output.mm(test)
        return grad_input, None

Simulate = Simulate.apply
print("Function built")

################################
#LOAD PRETRAINED MODEL
activeHiddenlayers = [280, 350]
class ActiveLearn(nn.Module):

    def __init__(self, n_in, out_sz):
        super(ActiveLearn, self).__init__()

        self.L_in = nn.Linear(n_in, activeHiddenlayers[0])
        self.H1 = nn.Linear(activeHiddenlayers[0], activeHiddenlayers[1])
        #self.H1 = nn.Linear(activeHiddenlayers[0], out_sz)
        self.H2 = nn.Linear(activeHiddenlayers[1], out_sz)
        self.L_out = nn.Linear(out_sz, out_sz)
        self.Relu = nn.ReLU(inplace=True)

    
    def forward(self, input):
        x = self.L_in(input)
        x = self.Relu(x)
        x = self.H1(x)
        x = self.Relu(x)
        x = self.H2(x)
        x = self.Relu(x)
        x = self.L_out(x)
        return x

model = ActiveLearn(input_size, output_size)
model.load_state_dict(torch.load(model_file_path_passive))
criterion = nn.SmoothL1Loss(reduction= 'sum')  # RMSE = np.sqrt(MSE)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size = 15, gamma=LRdecay, last_epoch=-1)
print("Model loaded")

####################################################
#TRAIN THE MODEL
torch.autograd.set_detect_anomaly(True)

start_time = time.time()
weight_c1 = 10 # basic error
weight_c2 = 1 # p start condition
weight_c3 = 1 # p smoothness condition
batch = np.floor(samplenum/minibatch_size).astype(int)
losses= []
smoothness_errors = []
p_start_errors = []
q_end_errors = [] #q_end error
for e in range(epochs):
    for b in range(batch):
        input_numpy = data[b*minibatch_size:b*minibatch_size+minibatch_size,:]
        input_tensor = torch.tensor(data[b*minibatch_size:b*minibatch_size+minibatch_size,:]).float()
        p_b = model(input_tensor)
        #print(p_b.shape)
        q_pred = Simulate(p_b, input_numpy)
        #print(p_b.shape)
        #smoothness_error_i = weight_c3*(p_i[0:3*(nTimeSteps-1)] - p_i[3:3*nTimeSteps]).pow(2).sum()
        smoothness_error = weight_c3*criterion(p_b[:, 0:dyn.nParameters*(nTimeSteps-1)], p_b[:, dyn.nParameters:dyn.nParameters*nTimeSteps])
        #p_start_error = weight_c2*torch.sqrt(criterion(p_i[0:3], torch.tensor(dyn.p_init[0:3])))
        p_start_error = weight_c2*criterion(p_b[:, 0:dyn.nParameters], input_tensor[:,21:25])
        #q_end_error = torch.sqrt(criterion(q_pred, q_i))
        q_end_error = weight_c1*criterion(q_pred[:, dyn.nDofs*(nTimeSteps-1):dyn.nDofs*(nTimeSteps-1)+3], input_tensor[:,0:3])
        #findnantorch(q_end_error)
        #findnantorch(p_start_error)
        #findnantorch(smoothness_error)
        loss = q_end_error + p_start_error + smoothness_error
        #print(loss)
        losses.append(loss)
        smoothness_errors.append(smoothness_error)
        q_end_errors.append(q_end_error)
        p_start_errors.append(p_start_error)
        optimizer.zero_grad()
        #Backpropagation
        loss.backward()
        optimizer.step()
    if e%(epochs/10) == 0:
        print(f'epoch: {e:3}/{epochs}  loss: {loss.item():10.8f}   basic_loss: {q_end_error.item():10.8f}')

print(f'epoch: {e:3} finale loss: {loss.item():10.8f}') # print the last line
print(f'\nDuration: {(time.time() - start_time)/60:.3f} min') # print the time elapsed

##################################################
#PLOT EVERY LOSS COMPONENT FOR EACH BATCH
timestr = time.strftime("%H%M")
# epoch_lines = np.arange(0, epochs, 1/batch)
epoch_i = np.arange(epochs*batch)/batch
plt.figure(figsize = [12,6])
loss = plt.plot(epoch_i, losses, label = 'total loss', linewidth=3)
q_end = plt.plot(epoch_i, q_end_errors, label = 'target error', linestyle='--')
smoothness = plt.plot(epoch_i, smoothness_errors, label = 'smoothness error', linestyle='--')
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
    input_example = input_tensor[0, :]
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
    input_example = input_tensor[0, :]
    traced_script_module = torch.jit.trace(model, input_example)

    # Test the torch script
    #test_input = torch.tensor([0, 2, 0.5])
    #original = model(test_input)
    #output_example = traced_script_module(test_input)

    traced_script_module.save(model_file_path + 'Serialized_Models/Serialized_model_active_' + use_case + f'_{nTimeSteps}tsteps_latest_{samplenum_target}.pt')
    traced_script_module.save(model_file_path + 'Serialized_Models/Serialized_model_active_' + use_case + f'_{nTimeSteps}tsteps_{samplenum}s_{minibatch_size}bs_{epochs}e_{LRdecay}lr_' + timestr + f'_{samplenum_target}.pt')
    print('Model saved')
