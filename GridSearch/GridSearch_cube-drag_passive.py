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
input_size = 25 # target (0:3), q (3:9), qdot (9:15), qddot (15:21), p_now (21:25)
samplenum_target = 10000
samplenum_target_test = 5000
use_case = 'cube-drag'
learning_rate = 0.001
LRdecay = 0.7
model_file_path = '../Trained_Models/'
sample_file_path = f'../Data/Samples/data_{use_case}_{nTimeSteps}tsteps_2315/'
simulation_file_path = '../Data/Simulations/cube-drag.sim'
objective_file_path = f'../Data/Objectives/cube-drag.obj'
# check dde version
print("using dde version: " + dde.__version__)
# set log level
dde.set_log_level(dde.LogLevel.off)
print(f'log level set to {dde.get_log_level()}')

print(os.listdir(sample_file_path))

dyn = dde.DynamicSequence()
dyn.loadFile(simulation_file_path, nTimeSteps)
output_size = dyn.nParameters*nTimeSteps

#########################################
#LOAD TRAINING SAMPLES
number_of_files = len(os.listdir(sample_file_path))-4
output_size = dyn.nParameters*nTimeSteps

with open(sample_file_path + f'data_0.json') as json_file:
    data = json.load(json_file)
    filesize = len(data['q_target'])
samplenum = filesize*number_of_files
p = np.zeros((samplenum, dyn.nParameters*nTimeSteps))
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
        for i, p_i in enumerate(data['p']):
            p[filenum*filesize+i, :] = np.array(p_i)

print(f'Shape of input: {input.shape}')
print(f'Shape of p: {p.shape}')
#Remove zeros
p = p[~(input == 0).all(1)]
input = input[~(input == 0).all(1)]

print(f'Shape of input after removing faulty samples: {input.shape}')
print(f'Shape of p after removing faulty samples: {p.shape}')

data = input[0:samplenum_target, :]
p = p[0:samplenum_target, :]
print(data.shape)
print(p.shape)

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

data[:, 15:18], extr_qddot_tran = minmaxscale(data[:, 15:18], None)
data[:, 18:21], extr_qddot_rot = minmaxscale(data[:, 18:21], None)

data = torch.tensor(data).float()
p = torch.tensor(p).float()

#########################################
#LOAD TEST SAMPLES
number_of_files_test = len(os.listdir(sample_file_path + 'data_test/'))

with open(sample_file_path + f'data_0.json') as json_file:
    data_ = json.load(json_file)
    filesize = len(data_['q_target'])
samplenum_test = filesize*number_of_files_test
p_test = np.zeros((samplenum, dyn.nParameters*nTimeSteps))
input_test = np.zeros((samplenum, input_size))
for filenum in range(number_of_files):
    with open(sample_file_path + f'data_{filenum}.json') as json_file:
        data_ = json.load(json_file)
        for i, q_target_i in enumerate(data_['q_target']):
            input_test[filenum*filesize+i, 0:3] = np.array(q_target_i)
        for i, q_i in enumerate(data_['q']):
            input_test[filenum*filesize+i, 3:9] = np.array(q_i)
            pmq = np.array([q_i[0],q_i[2], q_i[3], q_i[5]])
        for i, qdot_i in enumerate(data_['qdot']):
            input_test[filenum*filesize+i, 9:15] = np.array(qdot_i)
        for i, qddot_i in enumerate(data_['qddot']):
            input_test[filenum*filesize+i, 15:21] = np.array(qddot_i)
        for i, p_now_i in enumerate(data_['p_now']):
            input_test[filenum*filesize+i, 21:25] = np.array(p_now_i) - pmq
        for i, p_i in enumerate(data_['p']):
            p_test[filenum*filesize+i, :] = np.array(p_i)

print(f'\nShape of input_test: {input_test.shape}')
print(f'Shape of p_test: {p_test.shape}')
#Remove zeros
p_test = p_test[~(input_test == 0).all(1)]
input_test = input_test[~(input_test == 0).all(1)]

print(f'Shape of input_test after removing faulty samples: {input_test.shape}')
print(f'Shape of p_test after removing faulty samples: {p_test.shape}')

data_test = input_test[0:samplenum_target_test, :]
p_test = p_test[0:samplenum_target_test, :]
print(input_test.shape)
print(p_test.shape)

input_test[:, 15:18] = minmaxscale(input_test[:, 15:18], extr_qddot_tran)
input_test[:, 18:21] = minmaxscale(input_test[:, 18:21], extr_qddot_rot)

p_test = torch.tensor(p_test).float()

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
print("Function built")

#########################################
#Parameters
learning_rate = 0.001
epochs_s = [100, 200]
minibatch_size = [50]
LRdecay_s = [0.8]
hiddenlayers_s = [[280, 350, 300]] 
layerz = len(hiddenlayers_s[0])


#########################################
#GRIDSEARCH
timestr = time.strftime("%m%d%H")
start_time = time.time()
scores = []
index = 0
for e1 in epochs_s:
    for b1 in minibatch_size:
        for d in LRdecay_s:
            for h in hiddenlayers_s:
                class ActiveLearn(nn.Module):

                    def __init__(self, n_in, out_sz):
                        super(ActiveLearn, self).__init__()

                        self.L_in = nn.Linear(n_in, h[0])
                        self.H1 = nn.Linear(h[0], h[1])
                        self.H2 = nn.Linear(h[1], h[2])
                        self.H3 = nn.Linear(h[2], out_sz)
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
                        x = self.H3(x)
                        x = self.Relu(x)
                        x = self.L_out(x)
                        return x


                model = ActiveLearn(input_size, output_size)

                criterion = nn.SmoothL1Loss(reduction= 'sum') 
                #criterion = torch.nn.MSELoss(reduction = 'sum')
                optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
                scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size = 15, gamma=d, last_epoch=-1)
                #train
                torch.autograd.set_detect_anomaly(True)
                batch = np.floor(samplenum_target/b1).astype(int)
                losses= []
                p_smoothness_errors = []
                basic_errors = [] #y_end_ and p_start error
                for e in range(e1):
                    for b in range(batch):
                        input_i = data[b*b1:b*b1+b1,:]
                        p_i = p[b*b1:b*b1+b1,:]
                        p_pred = model(input_i)
                        #error claculation
                        loss_batch = criterion(p_pred, p_i)
                        losses.append(loss_batch)
                        optimizer.zero_grad()
                        #Back Prop
                        loss_batch.backward()
                        optimizer.step()
                    scheduler.step()
                    LR= scheduler.get_last_lr()
                    if e%(e1/10) == 0:
                        print(f'epoch: {e:3}/{e1}    LR: {LR[0]:10.6f}  loss: {loss_batch.item():10.8f}')
                    
                print(f'Model {index} trained')
                print(f'epoch: {e:3} final loss: {loss_batch.item():10.8f}') # print the last line
                print(f'Training completed. Total duration: {(time.time() - start_time)/60:.3f} min') # print the time elapsed

                #Test the data
                #model.eval()
                criterion2 = torch.nn.MSELoss(reduction = 'sum')
                input_testTensor = torch.tensor(data_test).float()
                losses_test= []
                with torch.no_grad():
                    for i in range(samplenum_target_test):
                        p_val = model(input_testTensor[i, :])
                        p_val_ = torch.unsqueeze(p_val, 0)
                        q_val = Simulate(p_val_, np.expand_dims(input_test[i, :], axis=0))
                        pmq = torch.tensor([input[i, 3], input[i, 5], input[i, 6], input[i, 8]])
                        p_start_error = criterion2(p_val[0:dyn.nParameters], input_testTensor[i,21:25] + pmq)
                        q_error = criterion2(q_val[0, dyn.nDofs*(nTimeSteps-1):dyn.nDofs*(nTimeSteps-1)+3], input_testTensor[i, 0:3])
                        smoothness_error = criterion2(p_val[0:dyn.nParameters*(nTimeSteps-1)], p_val[dyn.nParameters:dyn.nParameters*nTimeSteps])
                        loss2 = p_start_error + q_error + smoothness_error
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

with open(f'Gridsearch_scores_cube-drag_passive_{layerz}.json', 'w') as outfile:
    json.dump(scores, outfile)

