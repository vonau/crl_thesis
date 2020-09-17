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

########################################
#PARAMETERS
nTimeSteps = 60 #at 60 Hz
input_size = 9
samplenum_target = 15000
samplenum_target_test = 5000
use_case = 'qhold'
learning_rate = 0.001
LRdecay = 0.7
model_file_path = '../Trained_Models/'
sample_file_path = f'../Data/Samples/data_{use_case}_{nTimeSteps}tsteps_1024/'
simulation_file_path = '../Data/Simulations/pm_target.sim'
objective_file_path = f'..Data/Objectives/pm_qhold.obj'
# set log level
dde.set_log_level(dde.LogLevel.off)
print(os.listdir(sample_file_path))

dyn = dde.DynamicSequence()
dyn.loadFile(simulation_file_path, nTimeSteps)
output_size = dyn.nParameters*nTimeSteps

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
            input[filenum*filesize+i, 6:9] = np.array(p_now_i)

print(f'Shape of input: {input.shape}')
print(f'Shape of p: {p.shape}')
#Remove zeros
data = input[~(input == 0).all(1)]
p = p[~(p == 0).all(1)]
print(data.shape)
print(p.shape)

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#y_train, y_test, p_train, p_test = train_test_split(y_target, p, test_size = testsize)

#y_target = torch.tensor(y_train).float()
#p = torch.tensor(p_train).float()
p = torch.tensor(p).float()
#y_test = torch.tensor(y_test).float()
#p_test = torch.tensor(p_test).float()

#########################################
#LOAD TEST SAMPLES
number_of_files_test = len(os.listdir(sample_file_path + 'data_test/'))
samplenum_test = 1000*number_of_files_test

p_test = np.zeros((samplenum_test, 3*nTimeSteps))
input_test = np.zeros((samplenum_test, input_size))

for filenum in range(number_of_files_test):
    with open(sample_file_path + f'data_test/data_{filenum}.json') as json_file:
        data_ = json.load(json_file)
        filesize = len(data_['q'])
        for i, p_i in enumerate(data_['p']):
            p_test[filenum*filesize+i, :] = np.array(p_i)
        for i, q_i in enumerate(data_['q']):
            input_test[filenum*filesize+i, 0:3] = np.array(q_i)
        for i, qdot_i in enumerate(data_['qdot']):
            input_test[filenum*filesize+i, 3:6] = np.array(qdot_i)
        for i, p_now_i in enumerate(data_['p_now']):
            input_test[filenum*filesize+i, 6:9] = np.array(p_now_i)

print(f'Shape of input_test: {input_test.shape}')
print(f'Shape of p_test: {p_test.shape}')
#Remove zeros
input_test = input_test[~(input_test == 0).all(1)]
p_test = p_test[~(p_test == 0).all(1)]
print(input_test.shape)
print(p_test.shape)

input_test = torch.tensor(input_test).float()
p_test = torch.tensor(p_test).float()

#########################################
#Parameters
learning_rate = 0.001
epochs_s = [300]
minibatch_size = [30,50]
LRdecay_s = [0.5]
hiddenlayers_s = [[180, 140], [200, 200]] 
layerz = len(hiddenlayers_s)

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

start_time = time.time()
weight_c1 = 1 # q error
weight_c2 = 1 # p start condition
weight_c3 = 1 # p smoothness condition
weight_c4 = 100 # p smoothness condition
smoothness_errors_p = []
smoothness_errors_q = []
p_start_errors = []
tot_losses = []
for e1 in epochs_s:
    for b1 in minibatch_size:
        for d in LRdecay_s:
            for h in hiddenlayers_s:
                class ActiveLearn(nn.Module):

                    def __init__(self, n_in, out_sz):
                        super(ActiveLearn, self).__init__()

                        self.L_in = nn.Linear(n_in, h[0])
                        self.H1 = nn.Linear(h[0], h[1])
                        self.H2 = nn.Linear(h[1], out_sz)
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

                criterion = nn.SmoothL1Loss()  # RMSE = np.sqrt(MSE)
                optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
                scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size = 15, gamma=d, last_epoch=-1)
                #train
                torch.autograd.set_detect_anomaly(True)
                #weight_c1 = 1 # q error
                weight_c2 = 1 # p start condition
                weight_c3 = 1 # p smoothness condition
                weight_c4 = 100 # q smoothness condition
                batch = np.floor(samplenum/b1).astype(int)
                losses= []
                p_smoothness_errors = []
                basic_errors = [] #y_end_ and p_start error
                for e in range(e1):
                    for b in range(batch):
                        input_numpy = data[b*b1:b*b1+b1,:]
                        input_tensor = torch.tensor(data[b*b1:b*b1+b1,:], requires_grad = True).float()
                        p_b = model(input_tensor)
                        q_pred = Simulate(p_b, input_numpy)
                        # separate losses
                        smoothness_error_p = weight_c3*criterion(p_b[:, 0:dyn.nParameters*(nTimeSteps-1)], p_b[:, dyn.nParameters:dyn.nParameters*nTimeSteps])
                        smoothness_error_q = weight_c4*criterion(q_pred[:, 0:dyn.nDofs*(nTimeSteps-1)], q_pred[:, dyn.nDofs:dyn.nDofs*nTimeSteps])
                        p_start_error = weight_c2*criterion(p_b[:, 0:dyn.nParameters], input_tensor[:,6:9])
                        # sum up losses
                        loss = p_start_error + smoothness_error_p + smoothness_error_q
                        losses.append(loss)
                        smoothness_errors_p.append(smoothness_error_p)
                        smoothness_errors_q.append(smoothness_error_q)
                        p_start_errors.append(p_start_error)
                        optimizer.zero_grad()
                        #Backpropagation
                        loss.backward()
                        optimizer.step()
                    scheduler.step()
                    LR= scheduler.get_last_lr()
                    if e%(e1/10) == 0:
                        print(f'epoch: {e:3}/{e1}    LR: {LR[0]:10.6f}  loss: {loss.item():10.8f}')
                    
                print(f'Model {index} trained')
                print(f'epoch: {e:3} final loss: {loss.item():10.8f}') # print the last line
                print(f'Training completed. Total duration: {(time.time() - start_time)/60:.3f} min') # print the time elapsed


                #Test the data
                #model.eval()
                losses_test= []
                with torch.no_grad():
                    for i in range(samplenum_target_test):
                        p_val = model(input_test[i, :])
                        loss2 = criterion(p_val,p_test[i,:])
                        losses_test.append(loss2.clone().numpy())
                #plot test errors
                plot = plt.plot(losses, label = 'loss', linewidth=3)
                plt.legend()
                plt.yscale('log')
                plt.ylabel('error')
                plt.xlabel('batches')
                plt.savefig(f'../GridSearch_scores/{index}_Loss_.png')
                tot_error = sum(losses_test)
                mean_error = np.mean(losses_test)
                scores.append([index, tot_error, np.double(mean_error), e1, b1, d, h])
                index = index + 1
                print(f'TOTAL ERROR: {tot_error}    mean error: {mean_error}  epochs: {e1}    batchsize: {b1}   LRdecay: {d}  hiddenlayer{h}')
                print("\nNEXT MODEL")
print(f'\nDuration: {(time.time() - start_time)/60:.3f} min') # print the time elapsed

with open(f'Gridsearch_scores_qhold_active_{layerz}.json', 'w') as outfile:
    json.dump(scores, outfile)