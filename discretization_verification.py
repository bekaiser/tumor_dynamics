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
from utils import get_ghosts, gradient, laplacian


path = '/Users/bkaiser/Documents/data/tumor_invasion/'
output_path = path + 'output/'
figure_path = path + 'figures/'


# resolution
N = 400
#dt = 0.0005

#==============================================================================


Ne = 2.**(np.array([3,4,5,6,7,8])) #,9,10])) #,9,10,11,12,13,14]))
#Ne = np.array([512]) #2.**(np.array([3]))
err_1x = np.zeros(len(Ne))
err_1y = np.zeros(len(Ne))
err_2 = np.zeros(len(Ne))
#dxe = np.zeros(len(Ne))

for k in range(0,len(Ne)):


    N = int(Ne[k]+1) # cell edges

    x = np.linspace(0.,1.,num=N,endpoint=True) # cell edges
    y = np.copy(x)
    X,Y = np.meshgrid(x,y)
    dx = x[1]-x[0]
    dy = y[1]-y[0]


    constants = easydict.EasyDict({
    "N":N,
    "dx":dx,
    "dy":dy,
    })

    n = np.cos(np.pi*X)*np.cos(np.pi*Y)
    dnx = -np.pi*np.sin(np.pi*X)*np.cos(np.pi*Y)
    dny = -np.pi*np.cos(np.pi*X)*np.sin(np.pi*Y)
    d2n = -2.*np.pi**2.*np.cos(np.pi*X)*np.cos(np.pi*Y)
    ng = get_ghosts( n, constants )

    grad_nx, grad_ny = gradient( ng, constants )
    lap_n = laplacian( ng, constants )
    err_1x[k] = np.amax( abs(grad_nx-dnx) / np.amax(abs(dnx)) )
    err_1y[k] = np.amax( abs(grad_ny-dny) / np.amax(abs(dny)) )
    err_2[k] = np.amax( abs(lap_n-d2n) / np.amax(abs(d2n)) )


    plotname = figure_path +'/error_lap_%i.png' %N
    fig = plt.figure(figsize=(16, 4.5))
    plt.subplot(1,4,1)
    cs = plt.contourf(X,Y,np.abs(d2n-lap_n),200,cmap='gist_yarg') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.colorbar(cs)
    plt.title(r"$|$error$|$",fontsize=16)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,4,2)
    cs = plt.contourf(X,Y,d2n,200,cmap='seismic') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"$\nabla^2$ analytical",fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,4,3)
    cs = plt.contourf(X,Y,lap_n,200,cmap='seismic') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"$\nabla^2$ computed",fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,4,4)
    cs = plt.contourf(X,Y,n,200,cmap='seismic') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"$f(x,y)$",fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplots_adjust(top=0.925, bottom=0.12, left=0.05, right=0.975, hspace=0.3, wspace=0.3)
    plt.savefig(plotname,format="png"); plt.close(fig);

    plotname = figure_path +'/error_dx_%i.png' %N
    fig = plt.figure(figsize=(16, 5))
    plt.subplot(1,4,1)
    cs = plt.contourf(X,Y,np.abs(grad_nx-dnx),200,cmap='gist_yarg') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.colorbar(cs)
    plt.title(r"$|$error$|$",fontsize=16)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,4,2)
    cs = plt.contourf(X,Y,dnx,200,cmap='seismic') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"$\partial_x$ analytical",fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,4,3)
    cs = plt.contourf(X,Y,grad_nx,200,cmap='seismic') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"$\partial_x$ computed",fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,4,4)
    cs = plt.contourf(X,Y,n,200,cmap='seismic') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"$f(x,y)$",fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplots_adjust(top=0.925, bottom=0.12, left=0.05, right=0.975, hspace=0.3, wspace=0.3)
    plt.savefig(plotname,format="png"); plt.close(fig);

    plotname = figure_path +'/error_dy_%i.png' %N
    fig = plt.figure(figsize=(16, 5))
    plt.subplot(1,4,1)
    cs = plt.contourf(X,Y,np.abs(grad_ny-dny),200,cmap='gist_yarg') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.colorbar(cs)
    plt.title(r"$|$error$|$",fontsize=16)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,4,2)
    cs = plt.contourf(X,Y,dny,200,cmap='seismic') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"$\partial_y$ analytical",fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,4,3)
    cs = plt.contourf(X,Y,grad_ny,200,cmap='seismic') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"$\partial_y$ computed",fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,4,4)
    cs = plt.contourf(X,Y,n,200,cmap='seismic') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"$f(x,y)$",fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplots_adjust(top=0.925, bottom=0.12, left=0.05, right=0.975, hspace=0.3, wspace=0.3)
    plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'/error.png'
fig = plt.figure(figsize=(6, 5))
plt.subplot(1,1,1)
plt.loglog(Ne,err_1x,color='royalblue',linewidth=3) #,linestyle='None',marker='.')
#plt.loglog(Ne,err_1y,color='crimson',linewidth=3) #,linestyle='None',marker='.')
plt.loglog(Ne,err_2,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
plt.loglog(Ne,100*Ne**(-2.),color='black',linewidth=3)
#plt.legend(loc=2,fontsize=16,framealpha=1.)
plt.grid()
plt.ylabel(r"$||$error$||_2$",fontsize=16)
plt.xlabel(r"$N$ grid points",fontsize=16)
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.2, right=0.95, hspace=0.3, wspace=0.2)
plt.savefig(plotname,format="png"); plt.close(fig);
