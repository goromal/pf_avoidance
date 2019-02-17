#!/usr/bin/python3

from pf_avoidance import PotentialField as PF
import autograd.numpy as np

##############################################
################## TESTING ###################
##############################################

x = np.array([[1.0], [2.0], [1.0]]) # THESE NEED TO BE FLOATS!!!
s = np.array([[1.0], [1.0], [0.0]])

PF.addObstacle(1.0, 2, 3, 0, 0)
PF.addObstacle(0.0, 0, 0, 1, 1)

print(PF.Potential(x))
print(PF.Gradient(x))
print(PF.Hessian(x))
print(PF.directionalDerivative(x, s))
print(PF.secondDirectionalDerivative(x, s))
