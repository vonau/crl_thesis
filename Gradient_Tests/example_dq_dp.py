
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
p = np.zeros(dynSeq.p0.size*nTimeSteps)
for i in range(0,nTimeSteps):
	p[i*dynSeq.p0.size : (i+1)*dynSeq.p0.size] = dynSeq.p0

def dq_dp_fd_check(p):

	q = dynSeq.q(p)
	dq_dp = dynSeq.dq_dp(q, p)

	# check with FD
	fd = np.zeros((len(q.q), len(p)))
	h = 1e-8
	grad_output = np.ones([1, dynSeq.nDofs*nTimeSteps])
	for i in range(nTimeSteps*3):
		pp = np.copy(p)
		pp[i] = pp[i] + h
		pm = np.copy(p)
		pm[i] = pm[i] - h
		print("ym")
		print(dynSeq.q(pm).q)
		print("yp")
		print(dynSeq.q(pp).q)
		fd[:,i] = (dynSeq.q(pp).q - dynSeq.q(pm).q) / (2*h)

	analytical_grad= np.matmul(grad_output, dq_dp)
	numerical_grad = np.matmul(grad_output, fd)
	print("ANAL_GRAD:")
	print(analytical_grad)
	print("NUM_GRAD:")
	print(numerical_grad)
	err = LA.norm(analytical_grad - numerical_grad)
	print("error = {}".format(LA.norm(fd - dq_dp)))
	print(f'error of grad = {err}')

print(p)
#p[dq_dp_fd_check(p)
dq_dp_fd_check(p + 0.2)
#dq_dp_fd_check(p - 0.2)