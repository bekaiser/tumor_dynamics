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

from utils import load_nterms_angio, load_Enderling, get_restart_time

plot_flag = 'on'
data_set_name =  'tumor_angiogenesis' # 'tumor_invasion' #
path = '/Users/bkaiser/Documents/data/' + data_set_name + '/'
#path = '/Users/bkaiser/Documents/data/tumor_invasion/'
output_path = path + 'output/'
figure_path = path + 'figures/'
features_path = path + 'features/'

# grid resolution
N = 400

# time step:
q = 108200 #88600


#==============================================================================

if data_set_name == 'tumor_invasion':
    terms = load_Enderling( output_path, q )
    t = get_restart_time( q , output_path )
    print('time = ',t)

    # nterm data
    ddt = (terms.n_ddt).flatten('F')
    diff1 = (terms.n_diff1).flatten('F')
    diff2 = (terms.n_diff2).flatten('F')
    hapto1 = (terms.n_hapto1).flatten('F')
    hapto2 = (terms.n_hapto2).flatten('F')
    prolif = (terms.n_prolif).flatten('F')
    features = np.vstack([-ddt, diff1, diff2, hapto1, hapto2, prolif]).T

    print('max/min(ddt) = ',np.amax(ddt),np.amin(ddt))
    print('max/min(diff1) = ',np.amax(diff1),np.amin(diff1))
    print('max/min(diff2) = ',np.amax(diff2),np.amin(diff2))
    print('max/min(hapto1) = ',np.amax(hapto1),np.amin(hapto1))
    print('max/min(hapto2) = ',np.amax(hapto2),np.amin(hapto2))
    print('max/min(prolif) = ',np.amax(prolif),np.amin(prolif))
    print('max/min(features) = ',np.amax(features),np.amin(features))

elif data_set_name == 'tumor_angiogenesis':
    nterms = load_nterms_angio( output_path, q )
    t = get_restart_time( q , output_path )
    print('time = ',t)

    if plot_flag == 'on':
        from utils import load_fterms_angio, load_cterms_angio, load_vars_angio
        from plot_utils import plot_vars_angio, plot_vars_angio_noise, plot_all_terms_angio
        fterms = load_fterms_angio( output_path, q )
        cterms = load_cterms_angio( output_path, q )
        x = np.linspace(0.,1.,num=N,endpoint=True) # cell edges
        y = np.copy(x)
        X,Y = np.meshgrid(x,y)
        constants = easydict.EasyDict({
                    "N":N,
                    "x":x,
                    "y":y,
                    "X":X,
                    "Y":Y,
                    "figure_path":figure_path,
                    })
        plot_all_terms_angio( cterms, fterms, nterms, t, q, constants )
        c, f, n = load_vars_angio( output_path, q )
        plot_vars_angio( c, f, n, t, q, constants )

    # nterm data
    ddt = (nterms.ddt).flatten('F')
    diff = (nterms.diffusion).flatten('F')
    hapto1 = (nterms.hapto1).flatten('F')
    hapto2 = (nterms.hapto2).flatten('F')
    chemo1 = (nterms.chemo1).flatten('F')
    chemo2 = (nterms.chemo2).flatten('F')
    chemo3 = (nterms.chemo3).flatten('F')
    features = np.vstack([-ddt, diff, hapto1, hapto2, chemo1, chemo2, chemo3]).T

    print('max/min(ddt) = ',np.amax(ddt),np.amin(ddt))
    print('max/min(diff) = ',np.amax(diff),np.amin(diff))
    print('max/min(hapto1) = ',np.amax(hapto1),np.amin(hapto1))
    print('max/min(hapto2) = ',np.amax(hapto2),np.amin(hapto2))
    print('max/min(chemo1) = ',np.amax(chemo1),np.amin(chemo1))
    print('max/min(chemo2) = ',np.amax(chemo2),np.amin(chemo2))
    print('max/min(chemo3) = ',np.amax(chemo3),np.amin(chemo3))
    print('max/min(features) = ',np.amax(features),np.amin(features))


# write nterm data to empirical AI folder:
features_filename = features_path + 'features'
np.save(features_filename,features)

# x,y,area data
filename_x = output_path + 'x.npy'
x = np.load(filename_x)
filename_y = output_path + 'y.npy'
y = np.load(filename_y)
nx = len(x)
ny = len(y)
dx = x[1:nx]-x[0:nx-1]
dxe = np.append(dx,dx[0]) # grid is uniform in x
dy = y[1:ny]-y[0:ny-1]
yedges = np.append(0.,y[0:ny-1]+dy/2.)
ytop = y[ny-1]+dy[ny-2]/2.
yedges = np.append(yedges,ytop)
nedges = len(yedges)
dye = yedges[1:nedges]-yedges[0:nedges-1]
DXe,DYe = np.meshgrid(dxe,dye)
area = DXe*DYe

# write x,y,area data to empirical AI folder:
filename_x = features_path + 'x.npy'
np.save(filename_x,x)
filename_y = features_path + 'y.npy'
np.save(filename_y,y)
area_filename = features_path + 'area.npy'
np.save(area_filename,area)
