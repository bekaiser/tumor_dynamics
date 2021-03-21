import numpy as np
import math as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 15})
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

import matplotlib.colors as colors
import easydict

#==============================================================================
# functions

def plot_vars_invasion_noise( f, m, n, t, q, constants ):

    X = constants.X
    Y = constants.Y

    plotname = constants.figure_path +'noise_%i.png' %q
    fig = plt.figure(figsize=(16, 4.5))
    plt.subplot(1,3,1)
    #cs = contour_plot( f , constants )
    cs = plt.contourf(X,Y,f,100,cmap='inferno')
    plt.title(r"$f_{noise}(x,y,%.2f)$" %t,fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,3,2)
    #cs = contour_plot( m , constants )
    cs = plt.contourf(X,Y,m,100,cmap='inferno')
    plt.title(r"$m_{noise}(x,y,%.2f)$" %t,fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,3,3)
    #cs = contour_plot( n , constants )
    cs = plt.contourf(X,Y,n,100,cmap='inferno')
    plt.title(r"$n_{noise}(x,y,%.2f)$" %t,fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplots_adjust(top=0.925, bottom=0.15, left=0.075, right=0.95, hspace=0.3, wspace=0.3)
    plt.savefig(plotname,format="png"); plt.close(fig);

    return

def plot_vars_angio_noise( c, f, n, t, q, constants ):

    X = constants.X
    Y = constants.Y

    plotname = constants.figure_path +'noise_%i.png' %q
    fig = plt.figure(figsize=(16, 4.5))
    plt.subplot(1,3,1)
    #cs = contour_plot( f , constants )
    cs = plt.contourf(X,Y,c,100,cmap='inferno')
    plt.title(r"$c_{noise}(x,y,%.2f)$" %t,fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,3,2)
    #cs = contour_plot( m , constants )
    cs = plt.contourf(X,Y,f,100,cmap='inferno')
    plt.title(r"$f_{noise}(x,y,%.2f)$" %t,fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,3,3)
    #cs = contour_plot( n , constants )
    cs = plt.contourf(X,Y,n,100,cmap='inferno')
    plt.title(r"$n_{noise}(x,y,%.2f)$" %t,fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplots_adjust(top=0.925, bottom=0.15, left=0.075, right=0.95, hspace=0.3, wspace=0.3)
    plt.savefig(plotname,format="png"); plt.close(fig);

    return


def plot_vars_invasion( f, m, n, t, q, constants ):

    X = constants.X
    Y = constants.Y

    plotname = constants.figure_path +'vars_%i.png' %q
    fig = plt.figure(figsize=(16, 4.5))
    plt.subplot(1,3,1)
    #cs = contour_plot( f , constants )
    cs = plt.contourf(X,Y,f,100,cmap='inferno')
    plt.title(r"$f(x,y,%.2f)$" %t,fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,3,2)
    #cs = contour_plot( m , constants )
    cs = plt.contourf(X,Y,m,100,cmap='inferno')
    plt.title(r"$m(x,y,%.2f)$" %t,fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,3,3)
    #cs = contour_plot( n , constants )
    cs = plt.contourf(X,Y,n,100,cmap='inferno')
    plt.title(r"$n(x,y,%.2f)$" %t,fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplots_adjust(top=0.925, bottom=0.15, left=0.075, right=0.95, hspace=0.3, wspace=0.3)
    plt.savefig(plotname,format="png"); plt.close(fig);

    return

def plot_vars_angio( c, f, n, t, q, constants ):

    X = constants.X
    Y = constants.Y

    plotname = constants.figure_path +'vars_%i.png' %q
    fig = plt.figure(figsize=(16, 4.5))
    plt.subplot(1,3,1)
    cs = plt.contourf(X,Y,f,100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"$f(x,y,%.3f)$" %t,fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,3,2)
    cs = plt.contourf(X,Y,c,100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"$c(x,y,%.3f)$" %t,fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,3,3)
    cs = plt.contourf(X,Y,n,100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"$n(x,y,%.3f)$" %t,fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplots_adjust(top=0.925, bottom=0.15, left=0.075, right=0.95, hspace=0.3, wspace=0.3)
    plt.savefig(plotname,format="png"); plt.close(fig);

    return

def plot_mterms( mterms, t, q, constants ):
    # decay is off: sig = 0

    X = constants.X
    Y = constants.Y

    plotname = constants.figure_path +'mterms_%i.png' %q
    fig = plt.figure(figsize=(16, 4.5))
    plt.subplot(1,3,1)
    #cs = plt.contourf(X,Y,f/np.amax(f),100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    cs = plt.contourf(X,Y,mterms.ddt,100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"time rate of change: $\partial{m}/\partial{t}(x,y,%.4f)$" %t,fontsize=14)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,3,2)
    #cs = plt.contourf(X,Y,m/np.amax(m),100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    cs = plt.contourf(X,Y,mterms.diffusion,100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"diffusion: $d_m\nabla^2m(x,y,%.4f)$" %t,fontsize=14)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,3,3)
    #cs = plt.contourf(X,Y,n/np.amax(n),100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    cs = plt.contourf(X,Y,mterms.production,100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    #plt.plot(X[5,5+1],Y[5,5+1],color='red',marker='o',markersize=16)
    plt.title(r"production: $\kappa n(x,y,%.4f)$" %t,fontsize=14)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplots_adjust(top=0.925, bottom=0.15, left=0.075, right=0.95, hspace=0.3, wspace=0.3)
    plt.savefig(plotname,format="png"); plt.close(fig);

    return

def plot_fterms_angio( fterms, t, q, constants ):
    # decay is off: sig = 0

    X = constants.X
    Y = constants.Y

    plotname = constants.figure_path +'fterms_%i.png' %q
    fig = plt.figure(figsize=(16, 4.5))
    plt.subplot(1,3,1)
    #cs = plt.contourf(X,Y,f/np.amax(f),100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    cs = plt.contourf(X,Y,fterms.ddt,100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"time rate of change: $\partial{f}/\partial{t}(x,y,%.2f)$" %t,fontsize=14)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,3,2)
    cs = plt.contourf(X,Y,fterms.production,100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"production: $\beta n(x,y,%.2f)$" %t,fontsize=14)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(1,3,3)
    cs = plt.contourf(X,Y,fterms.uptake,100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"uptake: $-\gamma n f(x,y,%.2f)$" %t,fontsize=14)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplots_adjust(top=0.925, bottom=0.15, left=0.075, right=0.95, hspace=0.3, wspace=0.3)
    plt.savefig(plotname,format="png"); plt.close(fig);

    return

def plot_nterms( nterms, t, q, constants ):

    plotname = constants.figure_path +'nterms_%i.png' %q
    fig = plt.figure(figsize=(12, 12))
    plt.subplot(2,2,1)
    cs = contour_plot( nterms.ddt , constants ) #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"time rate of change: $\partial{n}/\partial{t}(x,y,%.4f)$" %t,fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(2,2,2)
    cs = contour_plot( nterms.diffusion , constants )
    plt.title(r"diffusion: $d_n\nabla^2n(x,y,%.4f)$" %t,fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(2,2,3)
    cs = contour_plot( nterms.haptotatic_diffusion , constants )
    plt.title(r"haptotaxis: $-\rho n\nabla^2{f}(x,y,%.4f)$" %t,fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplot(2,2,4)
    cs = contour_plot( nterms.haptotatic_dissipation , constants )
    plt.title(r"haptotaxis: $-\rho \nabla n \cdot \nabla{f}(x,y,%.4f)$" %t,fontsize=16)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplots_adjust(top=0.925, bottom=0.15, left=0.075, right=0.95, hspace=0.3, wspace=0.3)
    plt.savefig(plotname,format="png"); plt.close(fig);

    return

def plot_nterms_cterms_angio( nterms, cterms, t, q, constants ):

    ifontsize = 19
    ititlesize = 17
    plotname = constants.figure_path +'nterms_cterms_%i.png' %q
    fig = plt.figure(figsize=(24, 12))
    plt.subplot(2,4,1)
    cs = contour_plot( nterms.ddt , constants ) #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"time rate of change: $\partial{n}/\partial{t}(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)
    plt.subplot(2,4,2)
    cs = contour_plot( nterms.diffusion , constants )
    plt.title(r"diffusion: $D\nabla^2n(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)
    plt.subplot(2,4,3)
    cs = contour_plot( nterms.hapto1 , constants )
    plt.title(r"haptotaxis: $-\rho n\nabla^2{f}(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)
    plt.subplot(2,4,4)
    cs = contour_plot( nterms.hapto2 , constants )
    plt.title(r"haptotaxis: $-\rho \nabla n \cdot \nabla{f}(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)
    plt.subplot(2,4,5)
    cs = contour_plot( nterms.chemo1 , constants )
    plt.title(r"chemotaxis: $-\chi n \nabla^2{c}(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)
    plt.subplot(2,4,6)
    cs = contour_plot( nterms.chemo2 , constants )
    plt.title(r"chemotaxis: $-\chi \nabla n \cdot \nabla{c}(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)
    plt.subplot(2,4,7)
    cs = contour_plot( nterms.chemo3 , constants )
    plt.title(r"chemotaxis: $-n \nabla \chi \cdot \nabla{c}(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)

    plt.subplot(2,4,8)
    cs = contour_plot( cterms.ddt , constants )
    plt.title(r"time rate of change: $\partial{c}/\partial{t}(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)

    plt.subplots_adjust(top=0.96, bottom=0.075, left=0.04, right=0.975, hspace=0.25, wspace=0.2)
    plt.savefig(plotname,format="png"); plt.close(fig);

    return

def plot_all_terms_invasion( fterms, mterms, nterms, t, q, constants ):

    ifontsize = 19
    ititlesize = 17

    if constants.nonlinear_n_diff == 'on':

        plotname = constants.figure_path +'all_terms_%i.png' %q
        fig = plt.figure(figsize=(24, 18))
        plt.subplot(3,4,1)
        cs = contour_plot( nterms.ddt , constants ) #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
        plt.title(r"$\partial{n}/\partial{t}(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)
        plt.subplot(3,4,2)
        cs = contour_plot( nterms.diff1+nterms.diff2 , constants )
        plt.title(r"$\nabla\cdot(d_n m \nabla n)(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)
        plt.subplot(3,4,3)
        cs = contour_plot( nterms.hapto1+nterms.hapto2 , constants )
        plt.title(r"$-\nabla \cdot (\rho n \nabla f)(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)
        if constants.lam > 0:
            plt.subplot(3,4,4)
            cs = contour_plot( nterms.prolif , constants )
            plt.title(r"$\lambda n (1-n-f)(x,y,%.2f)$" %t,fontsize=ititlesize)
            plt.colorbar(cs)
            plt.ylabel(r"$y$",fontsize=ifontsize)
            plt.xlabel(r"$x$",fontsize=ifontsize)

        plt.subplot(3,4,5)
        cs = contour_plot( nterms.diff1 , constants )
        plt.title(r"$d_n\nabla m \cdot \nabla n(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)
        plt.subplot(3,4,6)
        cs = contour_plot( nterms.diff2 , constants )
        plt.title(r"$d_n m \nabla^2 n(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)
        plt.subplot(3,4,7)
        cs = contour_plot( nterms.hapto1 , constants )
        plt.title(r"$-\rho \nabla n \cdot \nabla f(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)
        plt.subplot(3,4,8)
        cs = contour_plot( nterms.hapto2 , constants )
        plt.title(r"$-\rho n \nabla^2 f(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)


        plt.subplot(3,4,9)
        cs = contour_plot( mterms.ddt , constants )
        plt.title(r"$\partial{m}/\partial{t}(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)
        plt.subplot(3,4,10)
        cs = contour_plot( mterms.diff , constants )
        plt.title(r"$d_m\nabla^2 m(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)
        plt.subplot(3,4,11)
        cs = contour_plot( mterms.prod , constants )
        if constants.nonlinear_m_production == 'on':
            plt.title(r"$\kappa n(1-m)(x,y,%.2f)$" %t,fontsize=ititlesize)
        else:
            plt.title(r"$\kappa n(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)
        plt.subplot(3,4,12)
        cs = contour_plot( fterms.ddt , constants )
        plt.title(r"$\partial{f}/\partial{t}(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=16)
        plt.xlabel(r"$x$",fontsize=16)

        plt.subplots_adjust(top=0.96, bottom=0.075, left=0.04, right=0.975, hspace=0.25, wspace=0.2)
        plt.savefig(plotname,format="png"); plt.close(fig);

    else:

        plotname = constants.figure_path +'all_terms_%i.png' %q
        fig = plt.figure(figsize=(18, 18))
        plt.subplot(3,3,1)
        cs = contour_plot( nterms.ddt , constants ) #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
        plt.title(r"$\partial{n}/\partial{t}(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)
        plt.subplot(3,3,2)
        cs = contour_plot( nterms.diff , constants )
        plt.title(r"$\nabla\cdot(d_n \nabla n)(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)
        plt.subplot(3,3,3)
        cs = contour_plot( nterms.hapto1+nterms.hapto2 , constants )
        plt.title(r"$-\nabla \cdot (\rho n \nabla f)(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)

        plt.subplot(3,3,4)
        cs = contour_plot( nterms.hapto1 , constants )
        plt.title(r"$-\rho \nabla n \cdot \nabla f(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)
        plt.subplot(3,3,5)
        cs = contour_plot( nterms.hapto2 , constants )
        plt.title(r"$-\rho n \nabla^2 f(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)
        plt.subplot(3,3,6)
        cs = contour_plot( fterms.ddt , constants )
        plt.title(r"$\partial{f}/\partial{t}(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=16)
        plt.xlabel(r"$x$",fontsize=16)

        plt.subplot(3,3,7)
        cs = contour_plot( mterms.ddt , constants )
        plt.title(r"$\partial{m}/\partial{t}(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)
        plt.subplot(3,3,8)
        cs = contour_plot( mterms.diff , constants )
        plt.title(r"$d_m\nabla^2 m(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)
        plt.subplot(3,3,9)
        cs = contour_plot( mterms.prod , constants )
        plt.title(r"$\kappa n(x,y,%.2f)$" %t,fontsize=ititlesize)
        plt.colorbar(cs)
        plt.ylabel(r"$y$",fontsize=ifontsize)
        plt.xlabel(r"$x$",fontsize=ifontsize)

        plt.subplots_adjust(top=0.96, bottom=0.075, left=0.04, right=0.975, hspace=0.25, wspace=0.2)
        plt.savefig(plotname,format="png"); plt.close(fig);

    return


def plot_all_terms_angio( cterms, fterms, nterms, t, q, constants ):

    ifontsize = 19
    ititlesize = 17
    plotname = constants.figure_path +'all_terms_%i.png' %q
    fig = plt.figure(figsize=(24, 18))
    plt.subplot(3,4,1)
    cs = contour_plot( nterms.ddt , constants ) #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"$\partial{n}/\partial{t}(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)
    plt.subplot(3,4,2)
    cs = contour_plot( nterms.diffusion , constants )
    plt.title(r"$D\nabla^2n(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)
    plt.subplot(3,4,3)
    # hapto1, hapto2 ~ grad(n) dot grad(f) , n laplacian(f)
    cs = contour_plot( nterms.hapto1 , constants )
    plt.title(r"$-\rho n \nabla^2{f}(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)
    plt.subplot(3,4,4)
    # hapto1, hapto2 ~ grad(n) dot grad(f) , n laplacian(f)
    cs = contour_plot( nterms.hapto2 , constants )
    plt.title(r"$-\rho\nabla{n}\cdot\nabla{f}(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)

    plt.subplot(3,4,5)
    cs = contour_plot( nterms.chemo1 , constants )
    plt.title(r"$-\chi n \nabla^2{c}(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)
    plt.subplot(3,4,6)
    cs = contour_plot( nterms.chemo2 , constants )
    plt.title(r"$-\chi\nabla{n}\cdot\nabla{c}(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)
    plt.subplot(3,4,7)
    cs = contour_plot( nterms.chemo3 , constants )
    plt.title(r"$-n \nabla\chi\cdot\nabla{c}(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)

    plt.subplot(3,4,9)
    cs = contour_plot( cterms.ddt , constants )
    plt.title(r"$\partial{c}/\partial{t}(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)
    plt.subplot(3,4,10)
    cs = contour_plot( fterms.ddt , constants )
    plt.title(r"$\partial{f}/\partial{t}(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)
    plt.subplot(3,4,11)
    cs = contour_plot( fterms.production , constants )
    plt.title(r"$\beta n(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=ifontsize)
    plt.xlabel(r"$x$",fontsize=ifontsize)
    plt.subplot(3,4,12)
    cs = contour_plot( fterms.uptake , constants )
    plt.title(r"$-\gamma n f(x,y,%.2f)$" %t,fontsize=ititlesize)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)

    plt.subplots_adjust(top=0.96, bottom=0.075, left=0.04, right=0.975, hspace=0.25, wspace=0.2)
    plt.savefig(plotname,format="png"); plt.close(fig);

    return


def plot_fterms( fterms, t, q, constants ):

    X = constants.X
    Y = constants.Y

    plotname = constants.figure_path +'fterms_%i.png' %q
    fig = plt.figure(figsize=(6, 5))
    plt.subplot(1,1,1)
    #cs = plt.contourf(X,Y,f/np.amax(f),100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    cs = plt.contourf(X,Y,fterms.ddt,100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"time rate of change: $\partial{f}/\partial{t}(x,y,%.4f)$" %t,fontsize=14)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplots_adjust(top=0.925, bottom=0.15, left=0.125, right=0.95, hspace=0.3, wspace=0.3)
    plt.savefig(plotname,format="png"); plt.close(fig);

    return

def plot_cterms_angio( cterms, t, q, constants ):

    X = constants.X
    Y = constants.Y

    plotname = constants.figure_path +'cterms_%i.png' %q
    fig = plt.figure(figsize=(6, 5))
    plt.subplot(1,1,1)
    #cs = plt.contourf(X,Y,f/np.amax(f),100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    cs = plt.contourf(X,Y,cterms.ddt,100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    plt.title(r"time rate of change: $\partial{c}/\partial{t}(x,y,%.4f)$" %t,fontsize=14)
    plt.colorbar(cs)
    plt.ylabel(r"$y$",fontsize=16)
    plt.xlabel(r"$x$",fontsize=16)
    plt.subplots_adjust(top=0.925, bottom=0.15, left=0.125, right=0.95, hspace=0.3, wspace=0.3)
    plt.savefig(plotname,format="png"); plt.close(fig);

    return

def contour_plot( var , constants ):

    X = constants.X
    Y = constants.Y
    cmapn = 'seismic'
    cmapn2 = 'inferno'
    mid_val = 0.

    vmin1 = np.amin(var)
    vmax1 = np.amax(var)
    midnorm = MidpointNormalize(vmin=vmin1, vcenter=0., vmax=vmax1)
    #cs = plt.contourf(X,Y,f/np.amax(f),100,cmap='inferno') #,color='goldenrod',linewidth=3) #,linestyle='None',marker='.')
    if abs( np.abs(vmax1)-np.abs(vmin1) ) > 4.*min( np.abs(vmax1), np.abs(vmin1) ):
        cs = plt.contourf(X,Y,var,100,cmap=cmapn2)
    else:
        cs = plt.contourf(X,Y,var,100,cmap=cmapn, norm=midnorm)

    return cs

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
