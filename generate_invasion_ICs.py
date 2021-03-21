import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 15})
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
import random

import easydict
from utils import smoothed_random_noise, sinusoidal_noise
from plot_utils import plot_vars_invasion, plot_vars_invasion_noise

path = '/home/bryan/Documents/data/tumor_invasion/'
output_path = path + 'output/'
figure_path = path + 'figures/'


N = 400 #1000 #
plot_flag = 'on'
#eps = 0.000625
#eps = 0.0025
eps = 0.01
xc = 0.5 # x coordinate, tumor center
yc = 0.5 # y coordinate, tumor center

# noise params:
iter = 0
noise_amp_f = 1.0
noise_nu_dt_f = 1e-8
noise_amp_m = 0.05
noise_nu_dt_m = 1e-8
noise_amp_n = 0.05
noise_nu_dt_n = 1e-8
noise_iterations_f = iter*5
noise_iterations_m = iter*5
noise_iterations_n = iter*5


# grid
x = np.linspace(0.,1.,num=N,endpoint=True) # cell edges
y = np.copy(x)
X,Y = np.meshgrid(x,y)
dx = x[1]-x[0]
dy = y[1]-y[0]

constants = easydict.EasyDict({
    "N":N,
    "dx":dx,
    "x":x,
    "y":y,
    "X":X,
    "Y":Y,
    "noise_amp_m":noise_amp_m,
    "noise_nu_dt_m":noise_nu_dt_m,
    "noise_iterations_m":noise_iterations_m,
    "noise_amp_f":noise_amp_f,
    "noise_nu_dt_f":noise_nu_dt_f,
    "noise_iterations_f":noise_iterations_f,
    "noise_amp_n":noise_amp_n,
    "noise_nu_dt_n":noise_nu_dt_n,
    "noise_iterations_n":noise_iterations_n,
    "output_path":output_path,
    "figure_path":figure_path,
    })


#===============================================================================
# noise

#noise_f = smoothed_random_noise( constants, 'f' , 'uniform' )
#noise_m = smoothed_random_noise( constants, 'm' , 'Gaussian sinusoidal' )
#noise_n = smoothed_random_noise( constants, 'n' , 'Gaussian sinusoidal' )
#noise_f = sinusoidal_noise( constants, 1600, 1.0 ) #140, 1.0 )
#noise_m = sinusoidal_noise( constants, 1600, 1.0 ) #70, 0.5 )
#noise_n = sinusoidal_noise( constants, 1600, 1.0 ) #35, 0.5 )
#noise_f = np.zeros([N,N])
#or i in range(0,N):
#    for j in range(0,N):
#        noise_f[i,j] = np.random.uniform(0.,1.,1)
noise_f = sinusoidal_noise( constants, 1000, 0.5 ) #140, 1.0 )
#noise_m = sinusoidal_noise( constants, 400, 0.25 ) #70, 0.5 )
noise_n = np.zeros([N,N]) #sinusoidal_noise( constants, 1200, 0.5 ) #35, 0.5 )
#noise_f = np.zeros([N,N])
noise_m = np.zeros([N,N])
#noise_n = np.zeros([N,N])
print('f noise max/min = ',np.amax(noise_f),np.amin(noise_f))
print('m noise max/min = ',np.amax(noise_m),np.amin(noise_m))
print('n noise max/min = ',np.amax(noise_n),np.amin(noise_n))
print()

#===============================================================================
# generate n

"""
# tumor density initial conditions:
n = np.zeros([N,N])
tht = np.pi/4.
sig2x = eps/2
sig2y = eps*4
a = np.cos(tht)**2./(2.*sig2x) + np.sin(tht)**2./(2.*sig2y)
b = - np.sin(2.*tht)**2./(4.*sig2x) + np.sin(2.*tht)**2./(4.*sig2y)
c = np.sin(tht)**2./(2.*sig2x) + np.cos(tht)**2./(2.*sig2y)
for i in range(0,N):
    for j in range(0,N):
        r = np.sqrt( (x[i]-xc)**2.+(y[j]-yc)**2. )
        #n[j,i] = np.exp(-r**2./eps) + noise_n[j,i]
        n[j,i] = np.exp(-r**2./eps)*(1.0+noise_n[j,i])
        #if n[j,i] >= 1.:
        #    n[j,i] = 1.
        if n[j,i] <= 0.:
            n[j,i] = 0.
"""
# tumor density initial conditions:
n = np.zeros([N,N])
tht = np.pi/4.
sig2x = eps/2
sig2y = eps*4
a = np.cos(tht)**2./(2.*sig2x) + np.sin(tht)**2./(2.*sig2y)
b = - np.sin(2.*tht)**2./(4.*sig2x) + np.sin(2.*tht)**2./(4.*sig2y)
c = np.sin(tht)**2./(2.*sig2x) + np.cos(tht)**2./(2.*sig2y)
for i in range(0,N):
    for j in range(0,N):
        r1 = np.sqrt( (x[i]-xc*0.9)**2.+(y[j]-yc*0.9)**2. )
        r2 = np.sqrt( (x[i]-xc*1.1)**2.+(y[j]-yc*1.1)**2. )
        r3 = np.sqrt( (x[i]-xc)**2.+(y[j]-yc)**2. )
        n[j,i] = np.exp(-r1**2./eps) + np.exp(-r2**2./(2.*eps)) # 1
        #n[j,i] = np.exp(-r3**2./eps) #+ np.exp(-r3**2./(1.5*eps))
n = n/np.amax(n)
print('n = ',np.amax(n),np.amin(n))


filename = output_path + 'n_0.npy'
np.save(filename,n)

#===============================================================================
# generate f

"""
f = np.zeros([N,N])
for i in range(0,N):
    for j in range(0,N):
        #f[j,i] = (1.-n[j,i]*0.5) + noise_f[j,i]
        r = np.sqrt( (x[i]-xc)**2.+(y[j]-yc)**2. )
        #f[j,i] = 1.+noise_f[j,i]-np.exp(-r**2./eps)*0.5 # np.exp(-r**2./eps) substituted for n[j,i]
        #f[j,i] = (1.-np.exp(-r**2./eps)*0.5)*(1.+noise_f[j,i]) # np.exp(-r**2./eps) substituted for n[j,i]
        #f[j,i] = (1.-n[j,i]/0.5) + noise_f[j,i]
        f[j,i] = (1.0-np.exp(-r**2./eps)*0.5)*5.0 + noise_f[j,i]
        #if f[j,i] >= 1.:
        #    f[j,i] = 1.
        if f[j,i] <= 0.:
            f[j,i] = 0.
#f = f/np.amax(f)
"""
f = np.zeros([N,N])
for i in range(0,N):
    for j in range(0,N):
        #f[j,i] = np.random.uniform(0.,1.,1) # 1
        f[j,i] = (1.-n[j,i]) + np.random.uniform(0.,1.,1)
#print('0.5*dx**2. = nu*dt = ',0.5*dx**2.)
f = f/np.amax(f)

filename = output_path + 'f_0.npy'
np.save(filename,f)


#===============================================================================
# generate m

"""
m = np.zeros([N,N])
for i in range(0,N):
    for j in range(0,N):
        r = np.sqrt( (x[i]-xc)**2.+(y[j]-yc)**2. )
        #m[j,i] = 0.5*np.exp(-r**2./eps) + noise_m[j,i] # np.exp(-r**2./eps) substituted for n[j,i]
        m[j,i] = 0.5*np.exp(-r**2./eps) #*(1.+noise_m[j,i])
        #m[j,i] = 0.5*n[j,i]*(1.+noise_m[j,i])
        #if m[j,i] >= 1.:
        #    m[j,i] = 1.
        if m[j,i] <= 0.:
            m[j,i] = 0.
"""
m = 0.5*n

filename = output_path + 'm_0.npy'
np.save(filename,m)


#===============================================================================
# plots

print('f max/min = ',np.amax(f),np.amin(f))
print('m max/min = ',np.amax(m),np.amin(m))
print('n max/min = ',np.amax(n),np.amin(n))

if plot_flag == 'on':
    plot_vars_invasion( f, m, n, 0., 0, constants )
    plot_vars_invasion_noise( noise_f, noise_m, noise_n, 0., 0, constants )

print('\nTumor invasion ICs generated!')
