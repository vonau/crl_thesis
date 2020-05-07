import numpy as np
import pydde as d
import json
import matplotlib.pyplot as plt

nTimeSteps = 60;

# state_init = dyn.compute(dyn.p_init)

# load simulation as dynamic sequence
dyn = d.PyDyn('../Data/rb-pendulum/twoRb.sim', nTimeSteps)

# load parameters
with open('../Data/rb-pendulum/twoRb.p') as f:
	data = json.load(f)
p = np.array(data['parameters'])

# compute trajectory
y = dyn.compute(p)

# plot x trajectory of 2nd rigid body
plt.plot(y.y[6::12])
plt.show()

# plot x velocity of 2nd rigid body
plt.plot(y.ydot[6::12])
plt.show()