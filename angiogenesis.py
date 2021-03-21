import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 15})
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

import easydict

from utils import rk4_angio, get_restart_time

path = '/Users/bkaiser/Documents/data/tumor_angiogenesis/'
output_path = path + 'output/'
figure_path = path + 'figures/'

# non-dimensional coefficients (Anderson and Chaplain 1998)
D = 0.00035
alpha = 0.6
chi0 = 0.38
rho = 0.34
beta = 0.05
gamma = 0.1
eta = 0.1

# simulation parameters:
N = 400
dt = 0.0000001 #(0.0005 Anderson 2005)
tf = 20
q0 = 0
plot_interval = 50
save_interval = 200
dt_tolerance = 0.005

#==============================================================================

if q0 != 0:
    t0 = get_restart_time( q0 , output_path )
else:
    t0 = 0.

# grid
x = np.linspace(0.,1.,num=N,endpoint=True) # cell edges
y = np.copy(x)
X,Y = np.meshgrid(x,y)
dx = x[1]-x[0]
dy = y[1]-y[0]
filename_x = output_path + '/x.npy'
np.save(filename_x,x)
filename_y = output_path + '/y.npy'
np.save(filename_y,y)


constants = easydict.EasyDict({
    "D": D,
    "alpha": alpha,
    "chi0": chi0,
    "rho": rho,
    "beta": beta,
    "eta": eta,
    "gamma": gamma,
    "t0":t0,
    "q0":q0,
    "N":N,
    "dt": dt,
    "dx": dx,
    "X": X,
    "Y": Y,
    "tf": tf,
    "plot_interval":plot_interval,
    "save_interval":save_interval,
    "output_path":output_path,
    "figure_path":figure_path,
    "dt_tolerance":dt_tolerance,
    })

fterms = easydict.EasyDict({
    "ddt": np.zeros([N,N]),
    "uptake": np.zeros([N,N]),
    "production": np.zeros([N,N]),
    })

nterms = easydict.EasyDict({
    "ddt": np.zeros([N,N]),
    "diffusion": np.zeros([N,N]),
    "hapto1": np.zeros([N,N]),
    "hapto2": np.zeros([N,N]),
    "chemo1": np.zeros([N,N]),
    "chemo2": np.zeros([N,N]),
    "chemo3": np.zeros([N,N]),
    })

cterms = easydict.EasyDict({
    "ddt": np.zeros([N,N]),
    })

# initial conditions
filename_c0 = output_path + 'c_%i.npy' %(q0)
c = np.load(filename_c0)
filename_f0 = output_path + 'f_%i.npy' %(q0)
f = np.load(filename_f0)
filename_n0 = output_path + 'n_%i.npy' %(q0)
n = np.load(filename_n0)


print('\n\n    Input data:')
print('    start time: ',t0)
print('    start time step: ',q0)
print('    ' + filename_c0)
print('    ' + filename_f0)
print('    ' + filename_n0)
print('    c max/min = ',np.amax(c),', ',np.amin(c))
print('    f max/min = ',np.amax(f),', ',np.amin(f))
print('    n max/min = ',np.amax(n),', ',np.amin(n))

c, f, n = rk4_angio( c, f, n, constants, cterms, fterms, nterms )
