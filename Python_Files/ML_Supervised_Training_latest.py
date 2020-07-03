# This script loads data and trains a model with this data. It then plots the errors during training and saves
# the model in three locations with three different states:
# - Serialized state for CPP applications with torch script
# - Saves the whole model
# - Saves just the state dict (weights)
# The last part of the filename is the current month and day.

########################################
#LIBRARIES
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
import pydde as d
import matplotlib.pyplot as plt
import json
import sklearn
import os

########################################
#PARAMETERS
time_length = 60; #seconds
epochs = 250
minibatch_size= 50
hiddenlayers = [100]
input_size = 3
output_size = 3*time_length
learning_rate = 0.01
LRdecay = 0.7
data_file_path = '../Data/Samples/data_20k_2x2x2/'
model_file_path = '../Trained_Model'
print(os.listdir(data_file_path))

#########################################
#LOAD SAMPLES
number_of_files = len(os.listdir(data_file_path))-2
number_of_samples = 1000*number_of_files

p = np.zeros((3*time_length, number_of_samples))
y_target = np.zeros((number_of_samples, 3))

for filenum in range(number_of_files):
    with open(data_file_path + f'data_{filenum}.json') as json_file:
        data = json.load(json_file)
        filesize = len(data['y_target'])
        for i, p_i in enumerate(data['parameter']):
            p[:, filenum*filesize+i] = np.array(p_i)
        for s, y_s in enumerate(data['y_target']):
            y_target[filenum*filesize+s, :] = np.array(y_s)
p = p.transpose()

print(f'Shape of y_target: {y_target.shape}')
print(f'Shape of p: {p.shape}')
#Remove zeros
#y_target = y_target[~(y_target == 0).all(1)]
#p = p[~(p == 0).all(1)]
#print(y_target.shape)
#print(p.shape)

y_target = torch.tensor(y_target).float()
p = torch.tensor(p).float()

################################
#BUILD MODEL
class PassiveLearn(nn.Module):

    def __init__(self, n_in, out_sz):
        super(PassiveLearn, self).__init__()

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

model = PassiveLearn(input_size, output_size)

criterion = nn.SmoothL1Loss()  # RMSE = np.sqrt(MSE)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size = 15, gamma=LRdecay, last_epoch=-1)

################################################
#TRAIN THE MODEL

torch.autograd.set_detect_anomaly(True)

start_time = time.time()
weight_c1 = 1 # p error condition
batch = np.floor(number_of_samples/minibatch_size).astype(int)
losses= []
p_smoothness_errors = []
basic_errors = [] #y_end_ and p_start error
for e in range(epochs):
    for b in range(batch):
        loss_batch = 0
        smoothness_error_batch = 0
        y_i = y_target[b*minibatch_size:b*minibatch_size+minibatch_size,:]
        p_i = p[b*minibatch_size:b*minibatch_size+minibatch_size,:]
        p_pred = model(y_i)

        #error claculation
        loss_batch = weight_c1* criterion(p_pred, p_i)
        losses.append(loss_batch)
        optimizer.zero_grad()
        #Back Prop
        loss_batch.backward()
        optimizer.step()
    scheduler.step()
    LR= scheduler.get_lr()
    if e%50 == 0:
        print(f'epoch: {e:3}/{epochs}    LR: {LR[0]:10.6f}  loss: {loss_batch.item():10.8f}')

print(f'epoch: {e:3} final loss: {loss_batch.item():10.8f}') # print the last line
print(f'\nTraining completed. Total duration: {(time.time() - start_time)/60:.3f} min') # print the time elapsed

##################################################
#PLOT LOSS FOR EACH BATCH AFTER EVERY EPOCHE
epoch_lines = np.arange(0, epochs*batch, batch)
print(epoch_lines.shape)
plt.figure(figsize = [12,6])
loss = plt.plot(losses, label = 'Loss')
plt.legend()
plt.yscale('log')
plt.ylabel('error')
plt.xlabel('batches')
for xc in epoch_lines:
    plt.axvline(x=xc, linewidth = 0.2)
plt.show()

#####################################################
#SAVE MODEL

import time
timestr = time.strftime("%m%d")

#Save entire Model
torch.save(model, '../Trained_Models/Model_020720_supervised_20k_samples.pt')
torch.save(model, '../Trained_Models/Model_latest.pt')

#Save parameters of Model
torch.save(model.state_dict(), '../Trained_Models/state_dict/Trained_Model_statedict_20000s_250e_07lr_' + timestr + '.pt')
torch.save(model.state_dict(), '../Trained_Models/state_dict/Model_statedict_latest.pt')

#Convert to Torch Script and save for CPP application
input_example = torch.tensor([0.5, 1.1, 0.5])
traced_script_module = torch.jit.trace(model, input_example)

# Test the torch script
#test_input = torch.tensor([0, 2, 0.5])
#original = model(test_input)
#output_example = traced_script_module(test_input)

traced_script_module.save("../Trained_Models/Serialized_Models/Serialized_model_latest.pt")
traced_script_module.save('../Trained_Models/Serialized_Models/Serialized_model_2x2x2_' + timestr + '.pt')