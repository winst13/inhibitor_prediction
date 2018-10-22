import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import fmin, fminbound
from scipy.signal import savgol_filter

'''
Gets the centered gradient of a given curve
Assumes that the points are evenly spaced 1 cycle apart
'''
def getGradient(curve):
    curve1 = np.append(curve, [0, 0])
    curve2 = np.append([0, 0], curve)
    diff = curve1 - curve2
    return diff/2

'''
Gets the Ct value given the curve
Only gets the first time that the curve crosses the threshold
cycle = some number of integer cycle numbers
curve = array of length cycle, each point corresponds to a point on the estimated curve
'''
def getCt(cycles, curve, threshold = 0.5):
    weighted_av = 0
    weighted_av += curve[0]
    norm_curve = curve/max(curve)
    for i in range(1, len(cycles)):
        if norm_curve[i] >= threshold and norm_curve[i] >= weighted_av:
            slope = (norm_curve[i]-norm_curve[i-1])/(cycles[i]-cycles[i-1])
            ct = (threshold - norm_curve[i-1])/slope + cycles[i-1]
            if ct <= cycles[i-1]:
                continue
            return ct
        else:
            weighted_av += norm_curve[i]
            weighted_av /= 2
    print ("Bad curve, never crosses threshold")
    print (norm_curve)
    return

'''
Uses a smooth second derivative max to find the Cp
Cp is more valuable than Ct, since we are using curves normalized to 1
    2nd derivative max remains constant, but threshold values shift
Uses 2nd degree interpolation on the 2nd derivative to find the max
'''
def getCp(cycles, curve):
    ddy = savgol_filter(curve, 5, 3, deriv=2)
    func = interp1d(cycles, ddy, kind='quadratic', fill_value='extrapolate')
    x_opt = fminbound(lambda x: -func(x), int(cycles[0]), int(cycles[-1]), disp=False)
    return x_opt
    