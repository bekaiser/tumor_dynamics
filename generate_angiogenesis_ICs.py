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
import h5py

import easydict
from utils import smoothed_random_noise, sinusoidal_noise
from plot_utils import plot_vars_angio, plot_vars_angio_noise

path = '/home/bryan/Documents/data/tumor_angiogenesis/'
output_path = path + 'output/'
figure_path = path + 'figures/'

# general params:
N = 400
plot_flag = 'on'
nu = (np.sqrt(5.)-0.1)/(np.sqrt(5.)-1.)
x0 = 1.0 # x coordinate, tumor center
y0 = 0.5 # y coordinate, tumor center
r0 = 0.1
eps1 = 0.45
eps2 = 0.001
f0 = 0.75

# noise params:
iter = 0
noise_amp_c = 0.05
noise_nu_dt_c = 1e-8
noise_amp_f = 1.0
noise_nu_dt_f = 1e-8
noise_amp_n = 0.05
noise_nu_dt_n = 1e-8
noise_iterations_c = iter*5
noise_iterations_f = iter*5
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
    "noise_amp_c":noise_amp_c,
    "noise_nu_dt_c":noise_nu_dt_c,
    "noise_iterations_c":noise_iterations_c,
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

noise_f = sinusoidal_noise( constants, 1800, 0.01 )
noise_c = sinusoidal_noise( constants, 1800, 0.01 )
#noise_c = np.zeros([N,N])
#noise_f = sinusoidal_noise( constants, 60, 0.25 ) #140, 1.0 )
#noise_c = np.zeros([N,N])
#for i in range(0,N):
#    for j in range(0,N):
#        noise_c[i,j] = np.random.uniform(-0.1,0.1,1)

noise_n = np.zeros([N,N])
print('c noise max/min = ',np.amax(noise_c),np.amin(noise_c))
print('f noise max/min = ',np.amax(noise_f),np.amin(noise_f))
print('n noise max/min = ',np.amax(noise_n),np.amin(noise_n))
print()

#===============================================================================


#===============================================================================
# generate Tumor Angiogenesis Factor (TAF) initial conditions (Anderson 1998)


# TAF
c = np.ones([N,N])
for i in range(0,N):
    for j in range(0,N):
        r = np.sqrt( (x[i]-x0)**2.+(y[j]-y0)**2. )
        if r >= 0.1:
            c[j,i] = ((nu-r)**2.)/((nu-r0)**2.)
        c[j,i] += noise_c[j,i]
        if c[j,i] >= 1.:
            c[j,i] = 1.
        elif c[j,i] <= 0.:
            c[j,i] = 0.
print('c max/min = ',np.amax(c),np.amin(c))
filename = output_path + 'c_0.npy'
np.save(filename,c)

#===============================================================================
# generate f (fibronection, a macromolecule)

f = np.zeros([N,N])
for i in range(0,N):
    for j in range(0,N):
        f[j,i] = f0*np.exp(-(x[i]**2.)/eps1)
        f[j,i] += noise_f[j,i]
        if f[j,i] >= 1.:
            f[j,i] = 1.
        elif f[j,i] <= 0.:
            f[j,i] = 0.
print('f max/min = ',np.amax(f),np.amin(f))
filename = output_path + 'f_0.npy'
np.save(filename,f)

#===============================================================================
# generate initial endothelial cell density distribution

n = np.zeros([N,N])
for i in range(0,N):
    for j in range(0,N):
        n[j,i] = np.exp(-(x[i]**2.)/eps2)*(np.sin(3.*np.pi*y[j])**2.)
        n[j,i] += noise_n[j,i]
        if n[j,i] >= 1.:
            n[j,i] = 1.
        elif n[j,i] <= 0.:
            n[j,i] = 0.

# scale the initial density to 1 cell per 5 micrometers^2 (Anderson and Chaplain 1998)
# based on the prescribed grid:
#n_scale = (N/200)**2.
#n = n/n_scale

print('n max/min = ',np.amax(n),np.amin(n))
filename = output_path + 'n_0.npy'
np.save(filename,n)


#===============================================================================
# plots

if plot_flag == 'on':
    plot_vars_angio(c, f, n, 0., 0, constants )
    plot_vars_angio_noise( noise_c, noise_f, noise_n, 0., 0, constants )

print('\nTumor angiogenesis ICs generated!')
