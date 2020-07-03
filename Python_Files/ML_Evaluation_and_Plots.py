# This script loads test data and a trained pytorch model. itthe makes the following plots:

# - LOSS OF EACH SAMPLE
# - RELATIVE ERROR AND ERROR FOR EACH AXIS OVER TIME
# - P_PRED AND P_TRUTH FOR SINGLE SAMPLE

# In the end an estimated p value with a random target is simulated to get a y and
# compared to the simulated y of the p retreived from trajectory optimization.

##########################################
#LOAD LIBRARIES
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import json
import pydde as d
import os

########################################
#PARAMETERS
time_length = 60 #seconds
hiddenlayers = [100]
samples_per_file = 1000
data_file_path = '../Data/Samples/data_20k_2x2x2/test_data/'
criterion = nn.SmoothL1Loss()  # Huber Loss
model_file_path = '../Trained_Models/Model_latest.pt'
model_statedict_file_path = '../Trained_Models/state_dict/Model_statedict_latest.pt'

#########################################
#LOAD SAMPLES
number_of_files = len(os.listdir(data_file_path))
number_of_samples = samples_per_file*number_of_files
p = np.zeros((3*time_length, number_of_samples))
y_target = np.zeros((number_of_samples, 3))
gradnorm_truth = np.zeros(number_of_samples)
iterations_truth = np.zeros(number_of_samples)
loss_truth  = np.zeros(number_of_samples)

for filenum in range(number_of_files):
    with open(data_file_path + f'data_{filenum}.json') as json_file:
        data = json.load(json_file)
        filesize = len(data['y_target'])
        for i, p_i in enumerate(data['parameter']):
            p[:, filenum*filesize+i] = np.array(p_i)
        for s, y_s in enumerate(data['y_target']):
            y_target[filenum*filesize+s, :] = np.array(y_s)
        for i, truth in enumerate(data['loss']):
            loss_truth[filenum*filesize+i] = np.array(truth)
        for i, truth in enumerate(data['iterations']):
            iterations_truth[filenum*filesize+i] = np.array(truth)
        for i, truth in enumerate(data['loss']):
            gradnorm_truth[filenum*filesize+i] = np.array(truth)
p = p.transpose()
print(f'Shape of y_target: {y_target.shape}')
print(f'Shape of p: {p.shape}')

y_test = torch.tensor(y_target).float()
p_test = torch.tensor(p).float()

#########################################
#LOAD MODEL
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

model = PassiveLearn(3, 180)
model.load_state_dict(torch.load(model_statedict_file_path))
#model = torch.load(model_file_path)

#########################################
#TEST THE DATA
losses_test= []
with torch.no_grad():
    for i in range(number_of_samples):
        p_val = model(y_test[i, :])
        loss = criterion(p_val,p_test[i,:])
        losses_test.append(loss.clone().numpy())#Test the data

#########################################
#PLOT LOSS OF EACH SAMPLE
plt.figure(figsize = [12,6])
loss_test = plt.plot(losses_test, label = 'loss')
plt.legend()
plt.ylabel('error')
plt.xlabel('sample')
plt.show()
tot_error = sum(losses_test)
print(tot_error)

#########################################
#CALCULATE RELATIVE AND ABSOLUTE ERROR OF EACH AXIS
rel_errors_norm = []
with torch.no_grad():
    for i in range(number_of_samples):
            p_val = model(y_test[i, :])
            p_truth = p_test[i,:]
            rel_error = np.linalg.norm((p_val - p_truth)/p_truth)
            rel_errors_norm.append(rel_error)

rel_errors_sum = torch.zeros(180)
abs_errors_sum = torch.zeros(180)
with torch.no_grad():
    for i in range(number_of_samples):
            p_val = model(y_test[i, :])
            p_truth = p_test[i,:]
            rel_error = (p_val - p_truth)/p_truth
            abs_error = np.abs(p_val - p_truth)
            rel_errors_sum = rel_errors_sum + rel_error
            abs_errors_sum = abs_errors_sum + abs_error
x_err_rel = rel_errors_sum[0::3]/(number_of_samples)
y_err_rel = rel_errors_sum[1::3]/(number_of_samples)
z_err_rel = rel_errors_sum[2::3]/(number_of_samples)

x_err = abs_errors_sum[0::3]/(number_of_samples)
y_err = abs_errors_sum[1::3]/(number_of_samples)
z_err = abs_errors_sum[2::3]/(number_of_samples)

#########################################
#PLOT RELATIVE ERROR AND ERROR FOR EACH AXIS OVER TIME
plt.figure(figsize = [10,10])
#plt.suptitle('Errors using 5000 samples', fontsize=16)

plt.subplot(2, 1, 2)
plt.plot(x_err_rel, label = 'x-axis')
plt.plot(y_err_rel, label = 'y-axis')
plt.plot(z_err_rel, label = 'z-axis')
plt.xlabel('time (ms)')
plt.title('relative error of each axis over time')
plt.legend()

plt.subplot(2, 1, 1)
plt.plot(x_err, label = 'x-axis')
plt.plot(y_err, label = 'y-axis')
plt.plot(z_err, label = 'z-axis')
plt.xlabel('time (ms)')
plt.title('mean absolute error of each axis over time')
plt.legend()
plt.show()

#############################################
#PLOT P_PRED AND P_TRUTH FOR SINGLE SAMPLE
randomsample = 9
with torch.no_grad():
        p_val = model(y_test[randomsample, :])
        p_truth = p_test[randomsample,:]
        x_val = p_val[0::3]
        y_val = p_val[1::3]
        z_val = p_val[2::3]
        x_truth = p_truth[0::3]
        y_truth = p_truth[1::3]
        z_truth = p_truth[2::3]
            

plt.figure(figsize = [12,6])
plt.plot(x_val, label = 'x-axis')
plt.plot(y_val, label = 'y-axis')
plt.plot(z_val, label = 'z-axis')
plt.plot(x_truth, '--', label = 'true x-axis')
plt.plot(y_truth, '--', label = 'true y-axis')
plt.plot(z_truth, '--', label = 'true z-axis')
plt.title('p_pred and p_truth for a single sample')
plt.xlabel('timestep')
plt.legend()
plt.show()

#####################################################
#TEST THE MODEL

# Generate simulation
dyn = d.PyDyn('../Data/point-mass_pendulum.sim', time_length)
state_init = dyn.compute(dyn.p_init)
f = dyn.f(state_init, dyn.p_init)
df = dyn.df_dp(state_init, dyn.p_init)
dy = dyn.dy_dp(state_init, dyn.p_init)

y_target_test_= torch.tensor([0.5, 1.5, 0.5])
p_ = model(y_target_test_)
y_target_ = y_target_test_.detach().numpy()
p_ = p_.detach().numpy()
p_truth_ = dyn.get_p(y_target_, dyn.p_init)

yTraj_test_ = dyn.compute(p_)
yTraj_truth_ = dyn.compute(p_truth_)

print('TEST OF THE MODEL')
print(f'\ntest for y_target:\n {y_target_test_}')
print(f'\nevaluated y_end:\n {yTraj_test_.y[-3:]}')
print(f'\nsimulated traj_opt y_end with p_truth:\n {yTraj_truth_.y[-3:]}')
print(f'\ndifference of y_end:\n {np.sum(yTraj_test_.y[-3:]-y_target_)}')
print(f'\nlast 6 entries of p predicted:\n {p_[-6:].transpose()}')
print(f'\nlast 6 entries of p from traj_opt:\n {p_truth_[-6:].transpose()}')