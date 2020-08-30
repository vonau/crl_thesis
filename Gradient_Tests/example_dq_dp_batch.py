
# if python doesn't find the pydde or dde lib, uncomment
# the following lines to see where python is looking for 
# library packages
# import sys
# print (sys.path)

import sys
import json

import numpy as np
from numpy import linalg as LA

import pydde
import time
import matplotlib.pyplot as plt

# help(pydde)
samplenum = 2
# check dde version
print("using dde version: " + pydde.__version__)

# set log level
pydde.set_log_level(pydde.LogLevel.off)
print(f'log level set to {pydde.get_log_level()}')

# load a sim as a dynamic sequence
nTimeSteps = 1
dynSeq = pydde.DynamicSequence()
# load from file
dynSeq.loadFile("pointmass.sim", nTimeSteps)

# make parameter trajectory
numParameters = dynSeq.p0.size
p_init = np.zeros(dynSeq.p0.size*nTimeSteps)
for i in range(0,nTimeSteps):
	p_init[i*dynSeq.p0.size : (i+1)*dynSeq.p0.size] = dynSeq.p0

#sample only p_init
p = np.ones((3*nTimeSteps, samplenum))
for i in range(samplenum):
    p[:,i] = p_init

def dq_dp_fd_check(p):

    dq_dp_batch = np.zeros([dynSeq.nDofs*nTimeSteps, dynSeq.nParameters*nTimeSteps])
    for i in range(samplenum):
        q = dynSeq.q(p[:,i])
        dq_dp = dynSeq.dq_dp(q, p[:,i])
        dq_dp_batch = dq_dp_batch + dq_dp
    dq_dp_batch = dq_dp_batch/samplenum

	# check with FD
    fd = np.zeros((len(q.q), len(p)))
    fd_batch = np.zeros((len(q.q), len(p)))
    h = 1e-8
    grad_output = np.ones([samplenum, dynSeq.nDofs*nTimeSteps])
    for s in range(samplenum):
        for i in range(nTimeSteps*3):
            pp = np.copy(p[:,s])
            pp[i] = pp[i] + h
            pm = np.copy(p[:,s])
            pm[i] = pm[i] - h
            print("ym")
            print(dynSeq.q(pm).q)
            print("yp")
            print(dynSeq.q(pp).q)
            fd[:,i] = (dynSeq.q(pp).q - dynSeq.q(pm).q) / (2*h)
        fd_batch = fd_batch + fd
    fd_batch = fd_batch/samplenum
        

    analytical_grad= np.matmul(grad_output, dq_dp_batch)
    numerical_grad = np.matmul(grad_output, fd_batch)
    print("ANAL_GRAD:")
    print(analytical_grad)
    print("NUM_GRAD:")
    print(numerical_grad)
    err = LA.norm(analytical_grad - numerical_grad)
    print("error = {}".format(LA.norm(fd_batch - dq_dp_batch)))
    print(f'error of grad = {err}')

print(p)
dq_dp_fd_check(p)
#dq_dp_fd_check(p + 0.2)
#dq_dp_fd_check(p - 0.2)