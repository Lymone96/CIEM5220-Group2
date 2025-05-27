'''
This file contains all the functions needed to run the ERS part of 
Part 2 of the SRE project
'''

##############################################################################
# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io import loadmat
import random


##############################################################################
'''function for reading the files'''

def fnc_read_induced_signals(file):
    # this file reads the matlab file that contains time signals from Groningen.
    # it will choose two signals based on variable 'file', which  must
    # have a number between 0 and 7. Finally, it will output the 
    # time vector, dt, and acceleration (which is in cm/s2)
    
    # load data
    Data = loadmat('Induced.mat')
    
    # choose data according to file
    var_names = ['FRB2', 'GARST', 'HKS', 'KANT', 'MID1', 'STDM', 'WINN', 'WSE']
    
    # load it
    data = Data[var_names[file]]
    
    # length of signal is NPTS
    NPTS = len(data[:,0])
    
    # dt is second entry in time vector
    Dt = data[1,0]
    
    # get all the signals: [hor1, hor2, vert]
    earth_signal = [data[:,1],data[:,2],data[:,3]]
    
    return NPTS, Dt, earth_signal

##############################################################################

def fnc_calc_covariance(ag_x, ag_y, ag_z, dt):
    i = 0
#     print(ag_x)
    while np.isnan(ag_x[i]).any()==False:
        i += 1
        if i == len(ag_x-1):
            i = i-1
            break
        
    
    ag_xx = np.power(ag_x[:i], 2)
    ag_yy = np.power(ag_y[:i], 2)
    ag_zz = np.power(ag_z[:i], 2)
    ag_xy = np.multiply(ag_x[:i], ag_y[:i])
    ag_yz = np.multiply(ag_y[:i], ag_z[:i])
    ag_zx = np.multiply(ag_z[:i], ag_x[:i])

    t = np.arange(0, len(ag_x[:i])*dt, dt)

    # Calculate the time independent Covariance matrix
    covar = np.zeros((3, 3))
    covar[0,0] = 1 / t[-1] * np.trapz(ag_xx, t)
    covar[0,1] = 1 / t[-1] * np.trapz(ag_xy, t)
    covar[0,2] = 1 / t[-1] * np.trapz(ag_zx, t)
    covar[1,0] = covar[0,1]
    covar[1,1] = 1 / t[-1] * np.trapz(ag_yy, t)
    covar[1,2] = 1 / t[-1] * np.trapz(ag_yz, t)
    covar[2,0] = covar[0,2]
    covar[2,1] = covar[1,2]
    covar[2,2] = 1 / t[-1] * np.trapz(ag_zz, t)

    # Calculate the Covariance in the principal directions, and puts the values
    # in descending order.
    covar_principal = np.sort(np.linalg.eig(covar)[0])[::-1]

    # Calculate the rotation angle theta
    theta = 1/2 * np.arctan2(2*covar[0,1], covar[0,0]-covar[1,1])

    while theta < 0:
        theta = theta + np.pi/2

    # Calculate the principal components of the horizontal ground motion 
    a_1 = np.cos(theta)*ag_x + np.sin(theta)*ag_y
    a_2 = -np.sin(theta)*ag_x + np.cos(theta)*ag_y

    PGA_x = np.abs(ag_x).max()
    PGA_y = np.abs(ag_y).max()
    PGA_z = np.abs(ag_z).max()
    PGA1 = np.abs(a_1).max()
    PGA2 = np.abs(a_2).max()
    PGA = max(PGA_x, PGA_y, PGA_z, PGA1, PGA2)

    if PGA1 > PGA2:
        ag_1 = a_1
        ag_2 = a_2
    else:
        ag_1 = a_2
        ag_2 = a_1

    theta = theta * 180 / np.pi
    return ag_1, ag_2


##############################################################################
'''Function that calculated the elastic response spectrum'''

def fnc_Elastic_Response_Spectrum(ag, dt, NPTS, xi):
    """
    This is a function that receives some input, calls the Newmark beta
    integration method for the calculation of the response of each SDoF 
    and calculates and stores the maximum value of the response of each SDoF
    system.

    Parameters:
    ag (numpy.ndarray): Ground acceleration in (g) [n,1].
    dt (float): Time interval between samples [1,1].
    NPTS (int): System samples [1,1].
    xi (float): System damping ratio [1,1].

    Returns:
    SA (numpy.ndarray): Elastic response spectrum in (g) for 201 different
                        SDoF systems (0s, 0.02s,...4s) [201,1].
    Fel (numpy.ndarray): Maximum force of the same elastic SDoF systems [201,1].
    """
    # For simplification of calculations we assume unit mass of the SDoF system
    m = 1
    
    # This is simply an index for the vectors SA and Fel
    i = 0
    
    SA = np.zeros(201)
    Fel = np.zeros(201)
    
    SA[0] = np.max(np.abs(ag))
    
    for Tn in np.arange(0.02, 4.02, 0.02):
        i += 1
        # Calls the function that calculates the response based on Newmarks beta
        # integration method
        a, f = fnc_Newmark_elastic(dt, NPTS, ag, m, Tn, xi)

        SA[i] = np.max(np.abs(a))
        Fel[i] = np.max(np.abs(f))
        
    return SA, Fel

##############################################################################
'''Newmark integration method'''


def fnc_Newmark_elastic(dt, NPTS, ag, m, T, xi):
    ''' 
    (c) TU Delft
    Newmark's Direct Integration Method for elastic system
     For more information about the Newmark beta integration method the
     student can look at "Dynamics of structures, Chopra, 4th edition pg. 174
     But this is out of the scope of this course, and that's why we are not 
     describing the script here!
    --------------------------------------------------------------------------
     Integrates a 1-DOF system with mass "m", spring stiffness "k" and damping
     coeffiecient "xi", subjected to ground acceleration.
     Returns the displacement, velocity and acceleration of the system with
     respect to an inertial frame of reference.
    
     SYNTAX
           [a,f] = fnc_Newmark_elastic(dt,NPTS,ag,m,T,xi)
    
     INPUT
           [dt] :        Time Interval between samples                   [1,1]
           [NPTS]:       System samples                                  [1,1]
           [ag] :        ground acceleration    in(g)                    [n,1]
           [m]:          System Mass                                     [1,1]
           [T]:          System Natural Period                           [1,1]
           [xi]:         System Damping ratio                            [1,1]
    
     OUTPUT
           [a]:        Acceleration response    in g      [n,1]
           [f]:        Force of the system                [n,1]
    
    ==========================================================================
    '''
    # First we define the integration coefficients of Newmark method
    gam = 1/2
    beta = 1/6
    #--------------------------------------------------------------------------
    # The acceleration time history should be converted from g to m/s^2
    ag = ag * 9.81 

    t = np.arange(NPTS)*dt

    wn = 2*np.pi/T
    c = 2*xi*wn*m
    k = wn**2*m
    p = - m*ag

    kgor = k + gam/(beta*dt)*c + m/(beta*dt**2)

    alpha = m/(beta*dt) + gam*c/beta
    b = 0.5*m/beta + dt*(0.5*gam/beta - 1)*c

    dp = np.diff(p)
    x = np.zeros_like(t)
    u = x.copy()
    a = x.copy()
    f = x.copy()
    x[0] = 0
    u[0] = 0
    a[0] = 1/m*(p[0]-k*x[0]-c*u[0])
    f[0] = 0
    for i in range(int(NPTS)-1):
        deltaP = dp[i] + alpha*u[i] + b*a[i]
        dx_i = deltaP/kgor
        du_i = gam/(beta*dt)*dx_i - gam/beta*u[i] + dt*(1-0.5*gam/beta)*a[i]
        da_i = 1/(beta*dt**2)*dx_i - 1/(beta*dt)*u[i] - 0.5/beta*a[i]
        x[i+1] = dx_i + x[i]
        u[i+1] = du_i + u[i]
        a[i+1] = da_i + a[i]
        f[i+1] = k * x[i+1]
    # We will convert the acceleration tim history response to g units again
    a = (a + ag)/9.81
    
    return a, f
