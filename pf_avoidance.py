#!/usr/bin/python3
"""@package PF Avoidance
Functions for avoiding obstacles dynamically in a path-following setting.

...
"""

from autograd import grad
import autograd.numpy as np

import warnings
warnings.filterwarnings("ignore") # for now...

def obstacleFunction(x, y, z, x_o, y_o, z_o, r_o, h_o):
    """Cost function for a single obstacle.

    @param x scalar position.
    @param y scalar position.
    @param z scalar position.
    """
    # TODO: CHANGE THIS
    return x**2 + y**2 + z**2 + r_o + h_o

class PotentialField(object):
    """Class for calculating potential function values and derivatives given
    obstacle and boundary positions.

    TODO: ADD VISUALIZATION TOOLS
    TODO: LOAD BOUNDARIES FROM FILE AND CALCULATE FORCES
    """
    _obstacle_info = []
    _f1 = grad(obstacleFunction, 0)
    _f2 = grad(obstacleFunction, 1)
    _f3 = grad(obstacleFunction, 2)
    _f11 = grad(_f1, 0)
    _f12 = grad(_f1, 1)
    _f13 = grad(_f1, 2)
    _f22 = grad(_f2, 1)
    _f23 = grad(_f2, 2)
    _f33 = grad(_f3, 2)

    @staticmethod
    def addObstacle(pn, pe, pd, r, h):
        """Add an obstacle to the potential function.

        @param pn Obstacle NORTH position (m)
        @param pe Obstacle EAST position (m)
        @param pd Obstacle DOWN position (m)
        @param r Obstacle radius (m)
        @param h Obstacle height (m)
        """
        PotentialField._obstacle_info.append([pn, pe, -pd, r, h])

    @staticmethod
    def addBoundaries(filename_or_data):
        """FUNCTIONALITY PENDING

        """
        pass

    @staticmethod
    def _accumulate(function_handle, x, accumulation):
        for obstacle in PotentialField._obstacle_info:
            x_o = obstacle[0]
            y_o = obstacle[1]
            z_o = obstacle[2]
            r_o = obstacle[3]
            h_o = obstacle[4]
            accumulation += function_handle(x, x_o, y_o, z_o, r_o, h_o)
        return accumulation

    @staticmethod
    def Potential(x):
        """Calculate potential at vector position x.

        @param x The 3x1 np.array position x
        """
        potential = 0
        return PotentialField._accumulate(PotentialField._potential, x, potential)

    @staticmethod
    def _potential(x, x_o, y_o, z_o, r_o, h_o):
        return obstacleFunction(x[0], x[1], x[2], x_o, y_o, z_o, r_o, h_o)[0]

    @staticmethod
    def Gradient(x):
        """Calculate gradient of the potential at vector position x.

        @param x The 3x1 np.array position x
        """
        gradient = np.zeros((3,1))
        return PotentialField._accumulate(PotentialField._gradient, x, gradient)

    @staticmethod
    def _gradient(x, x_o, y_o, z_o, r_o, h_o):
        g1 = PotentialField._f1(x[0], x[1], x[2], x_o, y_o, z_o, r_o, h_o)
        g2 = PotentialField._f2(x[0], x[1], x[2], x_o, y_o, z_o, r_o, h_o)
        g3 = PotentialField._f3(x[0], x[1], x[2], x_o, y_o, z_o, r_o, h_o)
        return np.array([g1, g2, g3])

    @staticmethod
    def Hessian(x):
        """Calculate Hessian of the potential at vector position x.

        @param x The 3x1 np.array position x
        """
        hessian = np.zeros((3, 3))
        return PotentialField._accumulate(PotentialField._hessian, x, hessian)

    @staticmethod
    def _hessian(x, x_o, y_o, z_o, r_o, h_o):
        f11 = PotentialField._f11(x[0], x[1], x[2], x_o, y_o, z_o, r_o, h_o)[0]
        f12 = PotentialField._f12(x[0], x[1], x[2], x_o, y_o, z_o, r_o, h_o)[0]
        f13 = PotentialField._f13(x[0], x[1], x[2], x_o, y_o, z_o, r_o, h_o)[0]
        f22 = PotentialField._f22(x[0], x[1], x[2], x_o, y_o, z_o, r_o, h_o)[0]
        f23 = PotentialField._f23(x[0], x[1], x[2], x_o, y_o, z_o, r_o, h_o)[0]
        f33 = PotentialField._f33(x[0], x[1], x[2], x_o, y_o, z_o, r_o, h_o)[0]
        return np.array([[f11, f12, f13],[f12, f22, f23],[f13, f23, f33]])

    @staticmethod
    def directionalDerivative(x, s):
        """Calculate (scalar) directional derivative of a function at vector
        position x and unit vector direction s.

        @param x The 3x1 np.array position x
        @param s The 3x1 np.array direction s
        """
        s = s / np.linalg.norm(s)
        g = PotentialField.Gradient(x)
        return np.dot(g.T, s)[0][0]

    @staticmethod
    def secondDirectionalDerivative(x, s):
        """Calculate (scalar) second directional derivative of a function at
        vector position x and unit vector direction s.

        @param x The 3x1 np.array position x
        @param s The 3x1 np.array direction s
        """
        s = s / np.linalg.norm(s)
        H = PotentialField.Hessian(x)
        g2 = np.dot(x.T * H, x)
        return np.dot(g2.T, s)[0][0]
