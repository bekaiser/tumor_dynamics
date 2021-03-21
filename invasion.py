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

from utils import get_restart_time, rk4_invasion

path = '/Users/bkaiser/Documents/data/tumor_invasion/'
output_path = path + 'output/'
figure_path = path + 'figures/'


# non-dimensional coefficients
"""
# (Anderson 2005)
dn = 0.0005
dm = 0.0005
rho = 0.01
eta = 50.
kap = 1.0
lam = 0.
nonlinear_m_production = 'off'
nonlinear_diff = 'off'

# (Anderson 2000)
dn = 0.001
dm = 0.001
rho = 0.005
eta = 10.
kap = 0.1
lam = 0.
nonlinear_m_production = 'off'
nonlinear_n_diff = 'off'

"""
# (Enderling 2007)
dn = 0.0001
dm = 0.0005
rho = 0.00005
eta = 10.
kap = 0.1 # alpha in Enderling et al. 2007
lam = 0.75 #0.75 # proliferation constant
nonlinear_m_production = 'on'
nonlinear_n_diff = 'on'


# simulation parameters:
N = 400
dt = 0.00000001 # initial dt
tf = 10.
q0 = 0
plot_interval = 50
save_interval = 200
dt_tolerance = 0.005


#==============================================================================

if q0 != 0:
    t0 = get_restart_time( q0 , output_path )
    print('t0 = ',t0)
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
    "dn": dn,
    "dm": dm,
    "rho": rho,
    "eta": eta,
    "kap": kap,
    "lam": lam,
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
    "nonlinear_n_diff":nonlinear_n_diff,
    "nonlinear_m_production":nonlinear_m_production,
    })

mterms = easydict.EasyDict({
    "ddt": np.zeros([N,N]),
    "diff": np.zeros([N,N]),
    "prod": np.zeros([N,N]),
    })

if nonlinear_n_diff == 'on':
    if lam > 0.:
        nterms = easydict.EasyDict({
            "ddt": np.zeros([N,N]),
            "diff1": np.zeros([N,N]),
            "diff2": np.zeros([N,N]),
            "hapto1": np.zeros([N,N]),
            "hapto2": np.zeros([N,N]),
            "prolif": np.zeros([N,N]),
            })
    else:
        nterms = easydict.EasyDict({
            "ddt": np.zeros([N,N]),
            "diff1": np.zeros([N,N]),
            "diff2": np.zeros([N,N]),
            "hapto1": np.zeros([N,N]),
            "hapto2": np.zeros([N,N]),
            })
else:
    if lam > 0.:
        nterms = easydict.EasyDict({
            "ddt": np.zeros([N,N]),
            "diff": np.zeros([N,N]),
            "hapto1": np.zeros([N,N]),
            "hapto2": np.zeros([N,N]),
            "prolif": np.zeros([N,N]),
            })
    else:
        nterms = easydict.EasyDict({
            "ddt": np.zeros([N,N]),
            "diff": np.zeros([N,N]),
            "hapto1": np.zeros([N,N]),
            "hapto2": np.zeros([N,N]),
            })

fterms = easydict.EasyDict({
    "ddt": np.zeros([N,N]),
    })

# initial conditions
filename_m0 = output_path + 'm_%i.npy' %q0
m = np.load(filename_m0)
filename_n0 = output_path + 'n_%i.npy' %q0
n = np.load(filename_n0)
filename_f0 = output_path + 'f_%i.npy' %q0
f = np.load(filename_f0)

print('\n\n    Input data:')
print('    start time: ',t0)
print('    start time step: ',q0)
print('    ' + filename_m0)
print('    ' + filename_n0)
print('    ' + filename_f0)
print('    f max/min = ',np.amax(f),', ',np.amin(f))
print('    m max/min = ',np.amax(m),', ',np.amin(m))
print('    n max/min = ',np.amax(n),', ',np.amin(n))

f, m, n = rk4_invasion( f, m, n, constants, fterms, mterms, nterms )
