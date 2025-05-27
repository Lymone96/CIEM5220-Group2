'''
This file contains all the functions needed to run the IERS part of 
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


##############################################################################
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
#         print('SA[i] = ' + str(np.max(np.abs(a))))
        SA[i] = np.max(np.abs(a))
        Fel[i] = np.max(np.abs(f))
        
    return SA, Fel

##############################################################################

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
#     print('t = ' +str(t))
    wn = 2*np.pi/T
    c = 2*xi*wn*m
    k = wn**2*m
    p = - m*ag

    kgor = k + gam/(beta*dt)*c + m/(beta*dt**2)

    alpha = m/(beta*dt) + gam*c/beta
    b = 0.5*m/beta + dt*(0.5*gam/beta - 1)*c

    dp = np.diff(p)
    x = np.zeros_like(t)
#     print('x = ' +str(x))
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

##############################################################################

def fnc_Direct_Inel_Resp_Spec(ag,dt,NPTS,xi,mu,SA_el,F_el):
    '''
    (c) TU Delft
    This is a function that calculates directly the constant ductility
    inelastic acceleration response spectrum based on the steps that have
    been described in the lecture 8
    ==========================================================================
    Main goal is for each SDoF system the capacity demand to be equal to 
    the predefined ductility capacity "mu_demand = mu".
    In order to specify how close these values we want to be, we have to
    introduce the acceptable "error"
    '''
    error = 0.0001
    
    # once more we assign a unit mass
    m = 1
    '''
    For SDoF system with natural period Tn=0s we assume that the response of 
    the elastic and the elastoplastic system are equal to PGA 
    "SA_in(Tn=0) = SA_el(Tn=0) = PGA", that the yield force is equal to
    the maximum force of the elastic system, thus the yield strength 
    reduction factor is unity "Ry(Tn=0)=1" and the ductility demand is equal 
    to the ductility capacity "mu_demand(Tn=0) = mu(Tn=0)".
    '''
    
    SA_in = np.zeros(len(SA_el))
    fy = np.zeros(len(SA_el))
    Ry = np.zeros(len(SA_el))
    mu_demand = np.zeros(len(SA_el))
    
    SA_in[0] = SA_el[0]
    fy[0] = F_el[0]
    Ry[0] = 1
    mu_demand[0] = mu
    Ry[1] = Ry[0]
    
    for i in range(1, 201):
#         print('i = ' +str(i))
        sas = 2
        Tn = (i)*0.02
        print('Calculating for Tn = ' +str(Tn))
        mu_demand[i] = 2 * mu
        
        while error < abs((mu - mu_demand[i])/mu):
#             print('Ry[i] = ' + str(Ry[i]))
            fy[i] = F_el[i] / Ry[i]
            uy = fy[i] / ((2*np.pi/Tn)**2*m)
            x, a = fnc_Newmark_elastoplastic(dt, NPTS, ag, m, Tn, xi, fy[i])
            SA_in[i] = np.max(np.abs(a))
            SD_in = np.max(np.abs(x))
            mu_demand[i] = SD_in / uy
            
            if mu > mu_demand[i]:
                Ry[i] = Ry[i] / ((1-1/(0.86*sas))**1.22)
            elif mu < mu_demand[i]:
                Ry[i] = Ry[i] * ((1-1/(0.86*sas))**1.22)
            sas = sas + 1
        
        Ry[i] = F_el[i] / fy[i]
        
        # guess the next Ry, but don't perform this if we are at the end of our loop
        if i != 200:
            Ry[i+1] = Ry[i]
    
    print('Done')
    
    return SA_in

##############################################################################

def fnc_Newmark_elastoplastic(dt, NPTS, ag, m, T, xi, fy):
    '''
     (c) TU Delft
     Newmark's Direct Integration Method for elastoplastic system
     For more information about the Newmark beta integration method the
     student can look at "Dynamics of structures, Chopra, 4th edition pg. 174
     But this is out of the scope of this course, and that's why we are not 
     describing the script here!
    --------------------------------------------------------------------------
     Integrates a 1-DOF elastoplastic system with mass "m", elastic spring 
     stiffness "k" and damping coeffiecient "xi", and yield force "fy" 
     subjected to ground acceleration.
     Returns the displacement, velocity and acceleration of the system with
     respect to an inertial frame of reference.

     SYNTAX
           [x,a] = fnc_Newmark_elastoplastic(dt,NPTS,ag,m,T,xi,fy)

     INPUT
           [dt] :        Time Interval between samples                   [1,1]
           [NPTS]:       System samples                                  [1,1]
           [ag] :        ground acceleration    in(g)                    [n,1]
           [m]:          System Mass                                     [1,1]
           [T]:          System Natural Period                           [1,1]
           [xi]:         System Damping ratio                            [1,1]
           [fy]:         System yield force                              [1,1]

     OUTPUT
           [x]:        Displacement response    in m      [n,1]
           [a]:        Acceleration response    in g      [n,1]

    ==========================================================================
    '''
    
    
    # this is for subdivision in case T is really small or Dt too big for
    # Newmark beta method
    NPTS_initial = NPTS
    Dt_initial = dt
    recordings_initial = ag
    T_el = T
    
    Nsub = np.ceil(Dt_initial*10/T_el).astype(int)
#     print('Nsub = ' +str(Nsub))
    if Nsub > 1:
        dt = Dt_initial / Nsub
#         print('NPTS_initial = ' +str(NPTS_initial))
        NPTS = (NPTS_initial-1)*Nsub + 1
#         print('NPTS = ' +str(NPTS))
        ag = np.zeros(NPTS)
        ag[0] = recordings_initial[0]
        i_sub = 0
        for ij in range(NPTS_initial-1):
            recordings_incr = (recordings_initial[ij+1] - recordings_initial[ij]) / Nsub
            for ji in range(Nsub):
                i_sub += 1
                ag[i_sub] = ag[i_sub-1] + recordings_incr
    else:
        ag = recordings_initial
        dt = Dt_initial
        NPTS = NPTS_initial
    
    #==========================================================================
    # First we define the integration coefficents of Newmark method
    gam = 1/2
    beta = 1/6

    ag = ag * 9.81
    t = np.arange(0, NPTS*dt, dt)
    wn = 2*np.pi/T
    c = 2*xi*wn*m
    k = wn**2*m
    p = -m*ag

    alpha1 = 1/(beta*dt**2)*m + gam/(beta*dt)*c
    alpha2 = 1/(beta*dt)*m + (gam/beta-1)*c
    alpha3 = (1/(2*beta) - 1)*m + dt*(gam/(2*beta)-1)*c

    x = np.zeros(len(t))
    u = np.zeros(len(t))
    a = np.zeros(len(t))
    fs = np.zeros(len(t))
    kT_hat = np.zeros(len(t))
    p_unchanged = np.zeros(len(t))
    kT = np.zeros(len(t))
    kT[0] = k*np.sign(p[0])
    x[0] = 0
    u[0] = 0
    a[0] = 1/m*(p[0]-k*x[0]-c*u[0])
    fs[0] = k*x[0]
    eR = 1e-3

         
    for i in range(NPTS-1):
        x[i+1] = x[i]
        fs[i+1] = fs[i]
        kT[i+1] = kT[i]
        p_unchanged[i+1] = p[i+1] + alpha1*x[i] + alpha2*u[i] + alpha3*a[i]
        deltaP = 1
        j=1
        while abs(deltaP) > eR:
            deltaP = p_unchanged[i+1] - fs[i+1] - alpha1*x[i+1]
            kT_hat[i+1] = kT[i+1] + alpha1
            dx_i = deltaP / kT_hat[i+1]
            x[i+1] = x[i+1] + dx_i
            fs[i+1] = fs[i] + k * ( x[i+1] - x[i] )
            if fs[i+1] >= fy:
                fs[i+1] = fy
                kT[i+1] = k
            if fs[i+1] <= -fy:
                fs[i+1] = -fy
                kT[i+1] = k
            j=j+1

        u[i+1] = gam/(beta*dt)*(x[i+1] - x[i]) + (1 - gam/beta)*u[i] + dt*(1 - gam/(2*beta))*a[i]
        a[i+1] = 1/(beta*dt**2)*(x[i+1] - x[i]) - 1/(beta*dt)*u[i] - (1/(2*beta) - 1)*a[i]

    for i in range(NPTS):
        # We will convert the acceleration time history response to g units again
        a[i] = (a[i] + ag[i])/9.81
        
    
    return x,a

##############################################################################

def fnc_tangent_stiff_elastoplastic(fs, k, fy, x):
    '''
    (c) TU Delft
    The stifness of the elastoplastic system is either k (elastic stiffness 
    of the system) or 0 when it is in the plastic region.
    The following script defines if we are in the elastic or the plastic
    region and returns the stifness "kT" and the force "fs" of the system at
    every time instant.
    =========================================================================
    '''
    if abs(fs) > fy:
        kT = 0
        fs = fy * np.sign(fs)
    else:
        kT = k * np.sign(fs * x)
    return kT, fs
