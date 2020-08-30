
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
nTimeSteps = 2
dynSeq = pydde.DynamicSequence()
# load from file
dynSeq.loadFile("../Data/Simulations/pointmass.sim", nTimeSteps)

# make parameter trajectory
numParameters = dynSeq.p0.size
p = np.zeros(dynSeq.p0.size*nTimeSteps)
for i in range(0,nTimeSteps):
	p[i*dynSeq.p0.size : (i+1)*dynSeq.p0.size] = dynSeq.p0

# run sim, this uses dynSeq.q0, dynSeq.qdot0, dynSeq.qddot0

# compute dq_dp
# print(dq_dp)

def dq_dp_fd_check(p):

	q = dynSeq.q(p)
	dq_dp = dynSeq.dq_dp(q, p)

	# check with FD
	fd = np.zeros((len(q.q), len(p)))
	h = 1e-8
	for i in range(0,6):
		pp = np.copy(p)
		pp[i] = pp[i] + h
		pm = np.copy(p)
		pm[i] = pm[i] - h
		fd[:,i] = (dynSeq.q(pp).q - dynSeq.q(pm).q) / (2*h)

	# print(dq_dp)
	# print(fd)

	print("error = {}".format(LA.norm(fd - dq_dp)))

print(p)
dq_dp_fd_check(p)
p = p + 0.2
print(p)
dq_dp_fd_check(p)

plt.show()
