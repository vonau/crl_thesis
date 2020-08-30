########################################
#LIBRARIES
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
import pydde as d
import json
import os
import time

########################################
#PARAMETERS
time_length = 60 #seconds
number_of_samples_train = 50
epochs = 20
minibatch_size= 10
hiddenlayers = [100]
input_size = 3
output_size = 3*time_length
learning_rate = 0.01
LRdecay = 0.7
model_file_path = 'Trained_Models'
data_file_path = 'Data/Samples/data_20k_2x2x2/'

#######################################
# LOAD SIMULATION
dyn = d.PyDyn('Data/point-mass_pendulum.sim', time_length)
#dyn = d.PyDyn('Data/rb-pendulum/twoRb.sim', time_length)
state_init = dyn.compute(dyn.p_init)
f = dyn.f(state_init, dyn.p_init)
df = dyn.df_dp(state_init, dyn.p_init)
dy = dyn.dy_dp(state_init, dyn.p_init)

##########################################
#SAMPLE TARGETS
y_target = np.zeros((number_of_samples_train,3))
y_target[:,2] = np.random.rand(number_of_samples_train)*2
y_target[:,1] = np.random.rand(number_of_samples_train)*2
y_target[:,0] = np.random.rand(number_of_samples_train)*2
y_target= torch.tensor(y_target).float()
p_start = torch.zeros((minibatch_size, 3)).float()
for i in range(minibatch_size):
    p_start[i, :] = torch.tensor(dyn.p_init[0:3])

##########################################
#BUILD CUSTOM SIMULATION FUNCTION
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
            state = dyn.compute(p[:, i])
            dy_dp = dyn.dy_dp(state, p[:, i])
            dy_dp = torch.tensor(dy_dp[-3:, :])
            dy_dp_batch = dy_dp_batch + dy_dp
        #print(f'dy/dp_batch: {dy_dp_batch/samplenum}')
        
        grad_input = torch.tensor(grad_output.mm(dy_dp_batch.float()/len(p[0,:])))
        #print(f'shape of grad input: {grad_input.shape}')
        #print(f'shape of grad output: {grad_output.shape}')
        return grad_input

Simulate = Simulate.apply

########################################
#BUILD CUSTOM MODEL
class ActiveLearn(nn.Module):

    def __init__(self, n_in, out_sz):
        super(ActiveLearn, self).__init__()

        self.L_in = nn.Linear(n_in, hiddenlayers[0])
        self.H1 = nn.Linear(hiddenlayers[0], 3*time_length)
        self.L_out = nn.Linear(3*time_length, 3*time_length)
        self.Relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.5)
    
    def forward(self, input):
        x = self.L_in(input)
        x = self.Relu(x)
        x = self.H1(x)
        x = self.Relu(x)
        x = self.L_out(x)
        return x

model = ActiveLearn(input_size, output_size)

criterion = nn.SmoothL1Loss()  # RMSE = np.sqrt(MSE)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size = 15, gamma=LRdecay, last_epoch=-1)

####################################################
#TRAIN THE MODEL
torch.autograd.set_detect_anomaly(True)

start_time = time.time()
weight_c1 = 2 # basic error
weight_c2 = 1 # p start condition
weight_c3 = 1 # p smoothness condition
batch = np.floor(number_of_samples_train/minibatch_size).astype(int)
losses= []
smoothness_errors = []
p_start_errors = []
y_end_errors = [] #y_end error
for e in range(epochs):
    for b in range(batch):
        y_b = y_target[b*minibatch_size:b*minibatch_size+minibatch_size,:].float()
        p_b = model(y_b)
        y_pred = Simulate(p_b)
        #smoothness_error_i = weight_c3*(p_i[0:3*(time_length-1)] - p_i[3:3*time_length]).pow(2).sum()
        smoothness_error = weight_c3*criterion(p_b[:, 0:3*(time_length-1)], p_b[:, 3:3*time_length])
        #p_start_error = weight_c2*torch.sqrt(criterion(p_i[0:3], torch.tensor(dyn.p_init[0:3])))
        p_start_error = weight_c2*criterion(p_b[:, 0:3], p_start)
        #y_end_error = torch.sqrt(criterion(y_pred.float(), y_i))
        y_end_error = weight_c1*criterion(y_pred, y_b)
        loss = y_end_error + p_start_error + smoothness_error
        #loss = loss_batch/minibatch_size
        losses.append(loss)
        #smoothness_error = smoothness_error_batch/minibatch_size
        smoothness_errors.append(smoothness_error)
        y_end_errors.append(y_end_error)
        p_start_errors.append(p_start_error)
        optimizer.zero_grad()
        #Backpropagation
        loss.backward()
        optimizer.step()
    if e%50 == 0:
        print(f'epoch: {e:3}/{epochs}  loss: {loss.item():10.8f}   basic_loss: {y_end_error.item():10.8f}')

print(f'epoch: {e:3} finale loss: {loss.item():10.8f}') # print the last line
print(f'\nDuration: {(time.time() - start_time)/60:.3f} min') # print the time elapsed

##################################################
#PLOT EVERY LOSS COMPONENT FOR EACH BATCH AND SAVE
timestr = time.strftime("%m%d")
epoch_lines = np.arange(0, epochs*batch, number_of_samples_train)
plt.figure(figsize = [12,6])
loss = plt.plot(losses, label = 'total loss', linewidth=3)
smoothness = plt.plot(smoothness_errors, label = 'smoothness error', linestyle='--')
y_end = plt.plot(y_end_errors, label = 'y_end error', linestyle='--')
p_start = plt.plot(p_start_errors, label = 'p_start error', linestyle='--')
plt.legend()
plt.yscale('log')
plt.ylabel('error')
plt.xlabel('batches')
for xc in epoch_lines:
    plt.axvline(x=xc, linewidth = 0.2)
plt.savefig('Plots/testfig_' + timestr + '.png')

#####################################################
#SAVE MODEL

timestr = time.strftime("%m%d")

#Save entire Model
torch.save(model, 'Trained_Models/Model_active_20k_samples_' + timestr + '.pt')
torch.save(model, 'Trained_Models/Model_active_latest.pt')

#Save parameters of Model
torch.save(model.state_dict(), f'Trained_Models/state_dict/Trained_Model_statedict_active_{number_of_samples_train}s_{epochs}e_{LRdecay}lr_' + timestr + '.pt')
torch.save(model.state_dict(), 'Trained_Models/state_dict/Model_statedict_active_latest.pt')

#Convert to Torch Script and save for CPP application
input_example = torch.tensor([0.5, 1.1, 0.5])
traced_script_module = torch.jit.trace(model, input_example)

# Test the torch script
#test_input = torch.tensor([0, 2, 0.5])
#original = model(test_input)
#output_example = traced_script_module(test_input)

traced_script_module.save("Trained_Models/Serialized_Models/Serialized_model_active_latest.pt")
traced_script_module.save('Trained_Models/Serialized_Models/Serialized_model_active_2x2x2_' + timestr + '.pt')