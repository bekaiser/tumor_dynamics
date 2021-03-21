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

def movement_probability( f, constants ):
    # path of an individual tumour cell predicted by probabilities of movement of
    # an individual cell in response to its local milieu.
    dt = constants.dt
    dx = constants.dx
    Dn = constants.Dn
    N = constants.N
    rho = constants.rho
    P0 = np.ones([N,N])*(1.-4.*dt*Dn/dx**2.)
    P1 = np.ones([N,N])*(dt*Dn/dx**2.)
    P2 = np.ones([N,N])*(dt*Dn/dx**2.)
    P3 = np.ones([N,N])*(dt*Dn/dx**2.)
    P4 = np.ones([N,N])*(dt*Dn/dx**2.)
    fg = get_ghosts( f, constants )
    for i in range(1,N+1):
        for j in range(1,N+1):
            P0[i-1,j-1] = P0[i-1,j-1] - dt*rho/dx**2.*(fg[i,j-1] + fg[i-1,j] - 4.0*fg[i,j] + fg[i+1,j] + fg[i,j+1])
            P1[i-1,j-1] = P1[i-1,j-1] - dt*rho/(4.*dx**2.)*(fg[i+1,j] - fg[i-1,j])
            P2[i-1,j-1] = P2[i-1,j-1] + dt*rho/(4.*dx**2.)*(fg[i+1,j] - fg[i-1,j])
            P3[i-1,j-1] = P3[i-1,j-1] - dt*rho/(4.*dx**2.)*(fg[i,j+1] - fg[i,j-1])
            P4[i-1,j-1] = P4[i-1,j-1] + dt*rho/(4.*dx**2.)*(fg[i,j+1] - fg[i,j-1])
    Ptotal = P0+P1+P2+P3+P4
    P0 = P0/Ptotal
    P1 = P1/Ptotal
    P2 = P2/Ptotal
    P3 = P3/Ptotal
    P4 = P4/Ptotal
    # upper bounds:
    R0 = P0
    R1 = P0+P1
    R2 = P0+P1+P2
    R3 = P0+P1+P2+P3
    R4 = np.ones([N,N])
    return R0,R1,R2,R3,R4

def tumor_cell_movement( itc, jtc, R0, R1, R2, R3, R4 ):
    # itc and jtc are the indices of the tumor cell
    R = np.random.uniform(0.,1.,1)[0]
    if R <= R0[itc,jtc]: # stationary:
        itc = itc
        jtc = jtc
    elif R <= R1[itc,jtc]: # move left:
        jtc = jtc-1
    elif R <= R2[itc,jtc]: # move right:
        jtc = jtc+1
    elif R <= R3[itc,jtc]: # move down:
        itc = itc-1
    elif R <= R4[itc,jtc]: # move up:
        itc = itc+1
    return itc,jtc

def get_ghosts( a, constants ):
    # array with ghost nodes for no-flux BCs:
    N = constants.N
    ag = np.zeros([N+2,N+2])
    ag[1:N+1,1:N+1] = a[:,:]
    #if order == '2nd':
    ag[0,1:N+1] = a[1,:] # second-order
    ag[1:N+1,0] = a[:,1]
    ag[N+1,1:N+1] = a[N-2,:]
    ag[1:N+1,N+1] = a[:,N-2]
    return ag

def divergence( a, ag, bg, constants ):
    # second-order divergence of a scalar dotted with the gradient of a scalar
    # calculates nabla dot ( a grad(b) ) = a laplacian(b) + grad(a) dot grad(b)
    # a laplacian(b) = "diff"
    # grad(a) dot grad(b) = "diss"
    grad_ax, grad_ay = gradient( ag, constants )
    grad_bx, grad_by = gradient( bg, constants )
    #div_a_db = np.multiply(grad_ax,grad_bx) + np.multiply(grad_ay,grad_by) + a * laplacian( bg, constants )
    #return div_a_db
    div1 = np.multiply(grad_ax,grad_bx) + np.multiply(grad_ay,grad_by)
    div2 = a * laplacian( bg, constants )
    return div1, div2 # grad(a) dot grad(b) ,  a laplacian(b)

def gradient( ag, constants ):
    # second-order gradient of scalar,  cell centered grid
    N = constants.N
    dxm1 = (constants.dx)**(-1.)
    grad_ax = np.zeros([N,N])
    grad_ay = np.zeros([N,N])
    for i in range(1,N+1):
        for j in range(1,N+1):
            grad_ay[i-1,j-1] = ( ag[i+1,j] - ag[i-1,j] ) * 0.5 * dxm1
            grad_ax[i-1,j-1] = ( ag[i,j+1] - ag[i,j-1] ) * 0.5 * dxm1
    return grad_ax, grad_ay

def laplacian( ag, constants ):
    # second-order Laplacian,  cell centered grid
    N = constants.N
    dxm2 = (constants.dx)**(-2.)
    lap_a = np.zeros([N,N])
    for i in range(1,N+1):
        for j in range(1,N+1):
            lap_a[i-1,j-1] = (ag[i,j-1] + ag[i-1,j] - 4.0*ag[i,j] + ag[i+1,j] + ag[i,j+1]) * dxm2
    return lap_a

def nRHS_invasion( f, m, n, constants ):
    # tumour cell motion equation right hand side
    N = constants.N
    dn = constants.dn
    rho = constants.rho
    lam = constants.lam
    ng = get_ghosts( n, constants )
    fg = get_ghosts( f, constants )
    if constants.nonlinear_n_diff == 'on':
        mg = get_ghosts( m, constants ) # nonlinear motility
        diff1, diff2 = divergence( m, mg, ng, constants ) # nonlinear motility
        diff1 = dn*diff1
        diff2 = dn*diff2
    else:
        diff = dn*laplacian( ng, constants )
    hapto1, hapto2 = divergence( n, ng, fg, constants )
    hapto1 = -rho*hapto1
    hapto2 = -rho*hapto2
    nRHS = easydict.EasyDict({
            "hapto1": hapto1,
            "hapto2": hapto2,
            })
    if constants.nonlinear_n_diff == 'on':
        if lam > 0.: # proliferation on
            prolif = lam*n*(np.ones([N,N])-n-f)
            nRHS.diff1 = diff1
            nRHS.diff2 = diff2
            nRHS.prolif = prolif
            nRHS.rk = diff1 + diff2 + hapto1 + hapto2 + prolif
            return nRHS
        else:
            nRHS.diff1 = diff1
            nRHS.diff2 = diff2
            nRHS.rk = diff1 + diff2 + hapto1 + hapto2
            return nRHS
    else:
        if lam > 0.: # proliferation on
            prolif = lam*n*(np.ones([N,N])-n-f)
            nRHS.diff = diff
            nRHS.prolif = prolif
            nRHS.rk = diff + hapto1 + hapto2 + prolif
            return nRHS
        else:
            nRHS.diff = diff
            nRHS.rk = diff + hapto1 + hapto2
            return nRHS

def fRHS_invasion( f, m, constants ):
    # extra-cellular matrix (ECM) equation right hand side
    eta = constants.eta
    fRHS = easydict.EasyDict({
        "rk": - eta*m*f,
        })
    return fRHS

def mRHS_invasion( m, n, constants ):
    # matrix-degradative enzyme (MDE) concentration equation right hand side
    # ...see 3.1.6 Production/degradation/diffusion (Anderson 2005)
    N = constants.N
    dm = constants.dm
    kap = constants.kap
    mg = get_ghosts( m, constants )
    diff = dm*laplacian( mg, constants )
    if constants.nonlinear_m_production == 'on':
        prod = kap*n*(np.ones([N,N])-m)
    else: # linear production
        prod = kap*n
    mRHS = easydict.EasyDict({
        "diff": diff,
        "prod": prod,
        "rk": diff + prod,
        })
    #km = diff + prod
    #return km, diff, prod
    return mRHS

def fRHS_angio( f, n, constants ):
    beta = constants.beta
    gamma = constants.gamma
    prodf = beta*n
    uptakef = - gamma*n*f
    kf = prodf + uptakef
    return kf, prodf, uptakef

def cRHS_angio( c, n, constants ):
    eta = constants.eta
    uptakec = - eta*c*n
    return uptakec

def nRHS_relax( n , ng, constants ):
    # this additional term enforces conservation of mass in the sense that
    # the density cannot be greater than unity at a grid point, so the accumulated
    # density must be removed.
    # second-order Laplacian,  cell centered grid
    factor  = constants.relax_factor # artificial diffusion
    n0  = constants.n0
    N = constants.N
    dxm2 = (constants.dx)**(-2.)
    relax = np.zeros([N,N])
    if factor == 0.:
        return relax
    else:
        for i in range(1,N+1):
            for j in range(1,N+1):
                if n[i-1,j-1] >= n0:
                    relax[i-1,j-1] = factor*(ng[i,j-1] + ng[i-1,j] - 4.0*ng[i,j] + ng[i+1,j] + ng[i,j+1]) * dxm2
        return relax

def nRHS_angio( c, f, n, constants ):
    N = constants.N
    D = constants.D
    rho = constants.rho
    chi0  = constants.chi0
    alpha  = constants.alpha
    chi = np.divide( np.ones([N,N])*chi0 , np.ones([N,N]) + alpha*c )
    ng = get_ghosts( n, constants )
    cg = get_ghosts( c, constants )
    fg = get_ghosts( f, constants )
    chig = get_ghosts( chi, constants )
    # equation terms:
    diffn = D*laplacian( ng, constants )
    gradn_dot_gradc, n_lapc = divergence( n, ng, cg, constants )
    grad_chix, grad_chiy = gradient( chig, constants )
    grad_cx, grad_cy = gradient( cg, constants )
    chemo_1 = - np.multiply( chi , n_lapc )
    chemo_2 = - np.multiply( chi , gradn_dot_gradc )
    chemo_3 = - np.multiply( n , np.multiply(grad_chix,grad_cx) + np.multiply(grad_chiy,grad_cy) )
    hapto_diss, hapto_diff = divergence( n, ng, fg, constants )
    hapto_1 = - rho * hapto_diff
    hapto_2 = - rho * hapto_diss
    kn = diffn + chemo_1 + chemo_2 + chemo_3 + hapto_1 + hapto_2
    return kn, diffn, chemo_1, chemo_2, chemo_3, hapto_1, hapto_2

def nRHS_angio_flux( c, f, n, constants ):
    # calculation in flux form:
    N = constants.N
    D = constants.D
    rho = constants.rho
    chi0  = constants.chi0
    alpha  = constants.alpha
    dxm1 = (constants.dx)**(-1.)
    chi = np.divide( np.ones([N,N])*chi0 , np.ones([N,N]) + alpha*c )
    ng = get_ghosts( n, constants )
    cg = get_ghosts( c, constants )
    fg = get_ghosts( f, constants )
    grad_cx, grad_cy = gradient( cg, constants )
    grad_fx, grad_fy = gradient( fg, constants )
    grad_nx, grad_ny = gradient( ng, constants )
    motility_flux_x = np.multiply(D,grad_nx)
    motility_flux_y = np.multiply(D,grad_ny)
    chemotaxis_flux_x = - np.multiply( chi, np.multiply(n,grad_cx) )
    chemotaxis_flux_y = - np.multiply( chi, np.multiply(n,grad_cy) )
    haptotaxis_flux_x = - rho*np.multiply(n,grad_fx)
    haptotaxis_flux_y = - rho*np.multiply(n,grad_fy)
    flux_x = motility_flux_x + chemotaxis_flux_x + haptotaxis_flux_x
    flux_y = motility_flux_y + chemotaxis_flux_y + haptotaxis_flux_y
    flux_xg = get_ghosts( flux_x, constants )
    flux_yg = get_ghosts( flux_y, constants )
    kn = np.zeros([N,N])
    for i in range(1,N+1):
        for j in range(1,N+1):
            kn[i-1,j-1] = ( flux_yg[i+1,j] - flux_yg[i-1,j] ) * 0.5 * dxm1
            + ( flux_xg[i,j+1] - flux_xg[i,j-1] ) * 0.5 * dxm1
    return kn

def cell_locations( constants, cells ):
    N = constants.N
    Ncells = cells.Ncells
    ic = cells.ic
    jc = cells.jc
    locs = np.zeros([N,N])
    for i in range(0,N):
        for j in range(0,N):
            for k in range(0,Ncells):
                if i == int(ic[k]) and j == int(jc[k]):
                    locs[i,j] = 1.
    cells.cell_locs = locs
    return cells

def rk4_invasion( f, m, n, constants, fterms, mterms, nterms ):

    from plot_utils import plot_vars_invasion, plot_all_terms_invasion

    N = constants.N
    dt = constants.dt
    tf = constants.tf
    plot_interval = constants.plot_interval
    save_interval = constants.save_interval
    f_sum = np.sum(np.sum(f))
    m_sum = np.sum(np.sum(m))
    n_sum = np.sum(np.sum(n))
    dt_tolerance = constants.dt_tolerance

    # compute solutions:
    t = constants.t0
    q = constants.q0
    time_data = np.array([])
    while t <= tf-dt:

        print('\n*********************************************')
        print('     step = ',q)
        print('     time = %.12f s' %(t))
        print('     dt = %.12f s' %(dt))

        if round(q/plot_interval) == (round(q)/plot_interval):
            print('     ...saving plots')
            plot_vars_invasion( f, m, n, t, q, constants )
            plot_all_terms_invasion( fterms, mterms, nterms, t, q, constants )

        if round(q/save_interval) == (round(q)/save_interval):
            print('     ...saving data')
            time_data = save_time( time_data, t, q, constants )
            save_vars_invasion( f, m, n, q, constants )
            save_all_terms_invasion( fterms, mterms, nterms, q, constants )

        # diffn1, diffn2 ~ nabla(m) dot nabla(n) , m laplacian(n)
        # hapto1, hapto2 ~ nabla(n) dot nabla(f) , n laplacian(f)

        f1 = fRHS_invasion( f, m, constants )
        m1 = mRHS_invasion( m, n, constants )
        n1 = nRHS_invasion( f, m, n, constants )

        f2 = fRHS_invasion( f + dt*f1.rk/2., m + dt*m1.rk/2., constants )
        m2 = mRHS_invasion( m + dt*m1.rk/2., n + dt*n1.rk/2., constants )
        n2 = nRHS_invasion( f + dt*f1.rk/2., m + dt*m1.rk/2., n + dt*n1.rk/2., constants )

        f3 = fRHS_invasion( f + dt*f2.rk/2., m + dt*m2.rk/2., constants )
        m3 = mRHS_invasion( m + dt*m2.rk/2., n + dt*n2.rk/2., constants )
        n3 = nRHS_invasion( f + dt*f2.rk/2., m + dt*m2.rk/2., n + dt*n2.rk/2., constants )

        f4 = fRHS_invasion( f + dt*f3.rk, m + dt*m3.rk, constants )
        m4 = mRHS_invasion( m + dt*m3.rk, n + dt*n3.rk, constants )
        n4 = nRHS_invasion( f + dt*f3.rk, m + dt*m3.rk, n + dt*n3.rk, constants )

        if dt_tolerance > 0.:

            dth = dt/2.

            f2h = fRHS_invasion( f + dth*f1.rk/2., m + dth*m1.rk/2., constants )
            m2h = mRHS_invasion( m + dth*m1.rk/2., n + dth*n1.rk/2., constants )
            n2h = nRHS_invasion( f + dth*f1.rk/2., m + dth*m1.rk/2., n + dth*n1.rk/2., constants )

            f3h = fRHS_invasion( f + dth*f2h.rk/2., m + dth*m2h.rk/2., constants )
            m3h = mRHS_invasion( m + dth*m2h.rk/2., n + dth*n2h.rk/2., constants )
            n3h = nRHS_invasion( f + dth*f2h.rk/2., m + dth*m2h.rk/2., n + dth*n2h.rk/2., constants )

            f4h = fRHS_invasion( f + dth*f3h.rk, m + dth*m3h.rk, constants )
            m4h = mRHS_invasion( m + dth*m3h.rk, n + dth*n3h.rk, constants )
            n4h = nRHS_invasion( f + dth*f3h.rk, m + dth*m3h.rk, n + dth*n3h.rk, constants )

        # update continuous variables
        kf = (f1.rk + 2.*f2.rk + 2.*f3.rk + f4.rk)/6.
        km = (m1.rk + 2.*m2.rk + 2.*m3.rk + m4.rk)/6.
        kn = (n1.rk + 2.*n2.rk + 2.*n3.rk + n4.rk)/6.
        if dt_tolerance > 0.:
            kfh = (f1.rk + 2.*f2h.rk + 2.*f3h.rk + f4h.rk)/6.
            kmh = (m1.rk + 2.*m2h.rk + 2.*m3h.rk + m4h.rk)/6.
            knh = (n1.rk + 2.*n2h.rk + 2.*n3h.rk + n4h.rk)/6.
            fh = f + dth*kfh/2.
            mh = m + dth*kmh/2.
            nh = n + dth*knh/2.
        f += dt*kf
        m += dt*km
        n += dt*kn

        # no negative values:
        f,m,n = no_negative_values(f,m,n,N=constants.N)
        if dt_tolerance > 0.:
            fh,mh,nh = no_negative_values(fh,mh,nh,N=constants.N)

        # adapt time step:
        if dt_tolerance > 0.:
            dt_new_f = modify_dt( f, fh, dt, dt_tolerance )
            dt_new_m = modify_dt( m, mh, dt, dt_tolerance )
            dt_new_n = modify_dt( n, nh, dt, dt_tolerance )
            dt = min(dt_new_f,dt_new_m,dt_new_n)

            # diffusion stability limit:
            if constants.nonlinear_n_diff == 'on':
                dt = min(dt,(constants.dx**2.)/(4*constants.dm),(constants.dx**2.)/(4*constants.dn*np.amax(m)))
            else:
                dt = min(dt,(constants.dx**2.)/(4*constants.dm),(constants.dx**2.)/(4*constants.dn))

            # minimum time step:
            dt = max(dt,1e-9)

        # m terms:
        mterms.ddt = km
        mterms.diff = (m1.diff + 2.*m2.diff + 2.*m3.diff + m4.diff)/6.
        mterms.prod = (m1.prod + 2.*m2.prod + 2.*m3.prod + m4.prod)/6.

        # n terms:
        nterms.ddt = kn
        if constants.nonlinear_n_diff == 'on':
            nterms.diff1 = (n1.diff1 + 2.*n2.diff1 + 2.*n3.diff1 + n4.diff1)/6. # nabla(m) dot nabla(n)
            nterms.diff2 = (n1.diff2 + 2.*n2.diff2 + 2.*n3.diff2 + n4.diff2)/6. # m laplacian(n)
        else:
            nterms.diff = (n1.diff + 2.*n2.diff + 2.*n3.diff + n4.diff)/6.
        nterms.hapto1 = (n1.hapto1 + 2.*n2.hapto1 + 2.*n3.hapto1 + n4.hapto1)/6. # nabla(n) dot nabla(f)
        nterms.hapto2 = (n1.hapto2 + 2.*n2.hapto2 + 2.*n3.hapto2 + n4.hapto2)/6. # n laplacian(f)
        if constants.lam > 0.:
            nterms.prolif = (n1.prolif + 2.*n2.prolif + 2.*n3.prolif + n4.prolif)/6. # n laplacian(f)

        # f terms:
        fterms.ddt = kf

        #print('     int(f/f0) = ',np.sum(np.sum(f))/f_sum)
        #print('     max/min(f) = ',np.amax(f),np.amin(f))
        print('     int(m/m0) = ',np.sum(np.sum(m))/m_sum)
        print('     max/min(m) = ',np.amax(m),np.amin(m))
        print('     int(n/n0) = ',np.sum(np.sum(n))/n_sum)
        print('     max/min(n) = ',np.amax(n),np.amin(n))

        if np.isnan(np.sum(np.sum(n))):
            print('ERROR: divergent simulation')
            return f, m, n

        t = t + dt
        q = q + 1

    return f, m, n

def save_all_terms_invasion( fterms, mterms, nterms, q, constants ):

    # f terms ******************************************************************
    filename_ddt = constants.output_path + 'f_ddt_%i.npy' %(q)
    np.save(filename_ddt,fterms.ddt)

    # m terms ******************************************************************
    filename_ddt = constants.output_path + 'm_ddt_%i.npy' %(q)
    filename_diffusion = constants.output_path + 'm_diff_%i.npy' %(q)
    filename_production = constants.output_path + 'm_prod_%i.npy' %(q)
    np.save(filename_ddt,mterms.ddt)
    np.save(filename_diffusion,mterms.diff)
    np.save(filename_production,mterms.prod)

    # n terms ******************************************************************
    if constants.nonlinear_n_diff == 'on':
        filename_diff1 = constants.output_path + 'n_diff1_%i.npy' %(q)
        filename_diff2 = constants.output_path + 'n_diff2_%i.npy' %(q)
        np.save(filename_diff1,nterms.diff1) # nabla(m) dot nabla(n)
        np.save(filename_diff2,nterms.diff2) # m laplacian(n)
    else:
        filename_diffusion = constants.output_path + 'n_diff_%i.npy' %(q)
        np.save(filename_diffusion,nterms.diff)
    filename_ddt = constants.output_path + 'n_ddt_%i.npy' %(q)
    np.save(filename_ddt,nterms.ddt)
    filename_hapto1 = constants.output_path + 'n_hapto1_%i.npy' %(q)
    filename_hapto2 = constants.output_path + 'n_hapto2_%i.npy' %(q)
    np.save(filename_hapto1,nterms.hapto1) # nabla(n) dot nabla(f)
    np.save(filename_hapto2,nterms.hapto2) # n laplacian(f)
    if constants.lam > 0.:
        filename_prolif = constants.output_path + 'n_prolif_%i.npy' %(q)
        np.save(filename_prolif,nterms.prolif) # nabla(n) dot nabla(f)
    return

def save_nterms_angio( nterms, t, q, constants ):

    # hapto1, hapto2 ~ grad(n) dot grad(f) , n laplacian(f)

    filename_ddt = constants.output_path + 'n_ddt_%i.npy' %(q)
    filename_diffusion = constants.output_path + 'n_diffusion_%i.npy' %(q)
    filename_hapto1 = constants.output_path + 'n_hapto1_%i.npy' %(q)
    filename_hapto2 = constants.output_path + 'n_hapto2_%i.npy' %(q)
    filename_chemo1 = constants.output_path + 'n_chemo1_%i.npy' %(q)
    filename_chemo2 = constants.output_path + 'n_chemo2_%i.npy' %(q)
    filename_chemo3 = constants.output_path + 'n_chemo3_%i.npy' %(q)
    np.save(filename_ddt,nterms.ddt)
    np.save(filename_diffusion,nterms.diffusion)
    np.save(filename_hapto1,nterms.hapto1)
    np.save(filename_hapto2,nterms.hapto2)
    np.save(filename_chemo1,nterms.chemo1)
    np.save(filename_chemo2,nterms.chemo2)
    np.save(filename_chemo3,nterms.chemo3)

    return

def save_mterms_invasion( mterms, q, constants ):

    filename_ddt = constants.output_path + 'm_ddt_%i.npy' %(q)
    filename_diffusion = constants.output_path + 'm_diff_%i.npy' %(q)
    filename_production = constants.output_path + 'm_prod_%i.npy' %(q)
    np.save(filename_ddt,mterms.ddt)
    np.save(filename_diffusion,mterms.diff)
    np.save(filename_production,mterms.prod)

    return

def save_fterms_angio( fterms, t, q, constants ):

    filename_ddt = constants.output_path + 'f_ddt_%i.npy' %(q)
    filename_uptake = constants.output_path + 'f_uptake_%i.npy' %(q)
    filename_production = constants.output_path + 'f_production_%i.npy' %(q)
    np.save(filename_ddt,fterms.ddt)
    np.save(filename_uptake,fterms.uptake)
    np.save(filename_production,fterms.production)

    return

def save_fterms_invasion( fterms, q, constants ):

    filename_ddt = constants.output_path + 'f_ddt_%i.npy' %(q)
    np.save(filename_ddt,fterms.ddt)

    return

def save_cterms_angio( cterms, t, q, constants ):

    filename_ddt = constants.output_path + 'c_ddt_%i.npy' %(q)
    np.save(filename_ddt,cterms.ddt)

    return

def save_vars_invasion( f, m, n, q, constants ):

    filename_f = constants.output_path + 'f_%i.npy' %(q)
    filename_m = constants.output_path + 'm_%i.npy' %(q)
    filename_n = constants.output_path + 'n_%i.npy' %(q)
    np.save(filename_f,f)
    np.save(filename_m,m)
    np.save(filename_n,n)

    return

def save_time( time_data, t, q, constants ):
    time_data = np.append(time_data,t)
    filename = constants.output_path + 't_%i.npy' %(q)
    np.save(filename,time_data)
    return time_data

def save_vars_angio( c, f, n, t, q, constants ):

    filename_f = constants.output_path + 'f_%i.npy' %(q)
    filename_c = constants.output_path + 'c_%i.npy' %(q)
    filename_n = constants.output_path + 'n_%i.npy' %(q)
    np.save(filename_f,f)
    np.save(filename_c,c)
    np.save(filename_n,n)

    return

def load_nterms_angio( output_path, q ):

    t0 = get_restart_time( q , output_path )
    #print('t = ',t0)

    # hapto1, hapto2 ~ grad(n) dot grad(f) , n laplacian(f)

    filename_ddt = output_path + 'n_ddt_%i.npy' %(q)
    filename_diffusion = output_path + 'n_diffusion_%i.npy' %(q)
    filename_hapto1 = output_path + 'n_hapto1_%i.npy' %(q)
    filename_hapto2 = output_path + 'n_hapto2_%i.npy' %(q)
    filename_chemo1 = output_path + 'n_chemo1_%i.npy' %(q)
    filename_chemo2 = output_path + 'n_chemo2_%i.npy' %(q)
    filename_chemo3 = output_path + 'n_chemo3_%i.npy' %(q)

    nterms = easydict.EasyDict({
           "ddt": np.load(filename_ddt),
           "diffusion": np.load(filename_diffusion),
           "hapto1": np.load(filename_hapto1),
           "hapto2": np.load(filename_hapto2),
           "chemo1": np.load(filename_chemo1),
           "chemo2": np.load(filename_chemo2),
           "chemo3": np.load(filename_chemo3),
           "t": t0,
           "q": q,
           })

    return nterms


def load_Enderling( output_path, q ):

    t0 = get_restart_time( q , output_path )
    print('t = ',t0)

    filename_ddt = output_path + 'n_ddt_%i.npy' %(q)
    filename_diff1 = output_path + 'n_diff1_%i.npy' %(q)
    filename_diff2 = output_path + 'n_diff2_%i.npy' %(q)
    filename_hapto1 = output_path + 'n_hapto1_%i.npy' %(q)
    filename_hapto2 = output_path + 'n_hapto2_%i.npy' %(q)
    filename_prolif = output_path + 'n_prolif_%i.npy' %(q)

    terms = easydict.EasyDict({
           "n_ddt": np.load(filename_ddt),
           "n_diff1": np.load(filename_diff1),
           "n_diff2": np.load(filename_diff2),
           "n_hapto1": np.load(filename_hapto1),
           "n_hapto2": np.load(filename_hapto2),
           "n_prolif": np.load(filename_prolif),
           "t": t0,
           "q": q,
           })

    return terms

def load_nterms( output_path, q ):

    filename_ddt = output_path + 'n_ddt_%i.npy' %(q)
    filename_diffusion = output_path + 'n_diffusion_%i.npy' %(q)
    filename_haptotatic_diffusion = output_path + 'n_haptotatic_diffusion_%i.npy' %(q)
    filename_haptotatic_dissipation = output_path + 'n_haptotatic_dissipation_%i.npy' %(q)

    nterms = easydict.EasyDict({
           "ddt": np.load(filename_ddt),
           "diffusion": np.load(filename_diffusion),
           "haptotatic_diffusion": np.load(filename_haptotatic_diffusion),
           "haptotatic_dissipation": np.load(filename_haptotatic_dissipation),
           })

    return nterms

def load_mterms( output_path, q ):

    filename_ddt = output_path + 'm_ddt_%i.npy' %(q)
    filename_diffusion = output_path + 'm_diffusion_%i.npy' %(q)
    filename_production = output_path + 'm_production_%i.npy' %(q)

    mterms = easydict.EasyDict({
        "ddt": np.load(filename_ddt),
        "diffusion": np.load(filename_diffusion),
        "production": np.load(filename_production),
        })

    return mterms

def load_fterms( output_path, q ):

    filename_ddt = output_path + 'f_ddt_%i.npy' %(q)

    fterms = easydict.EasyDict({
        "ddt": np.load(filename_ddt),
        })

    return fterms

def load_cterms_angio( output_path, q ):

    filename_ddt = output_path + 'c_ddt_%i.npy' %(q)

    cterms = easydict.EasyDict({
        "ddt": np.load(filename_ddt),
        })

    return cterms

def load_fterms_angio( output_path, q ):

    filename_ddt = output_path + 'f_ddt_%i.npy' %(q)
    filename_production = output_path + 'f_production_%i.npy' %(q)
    filename_uptake = output_path + 'f_uptake_%i.npy' %(q)

    fterms = easydict.EasyDict({
        "ddt": np.load(filename_ddt),
        "production": np.load(filename_production),
        "uptake": np.load(filename_uptake),
        })

    return fterms

def load_vars( output_path, q ):

    filename_f = output_path + 'f_%i.npy' %(q)
    filename_m = output_path + 'm_%i.npy' %(q)
    filename_n = output_path + 'n_%i.npy' %(q)

    f = np.load(filename_f)
    m = np.load(filename_m)
    n = np.load(filename_n)

    return f, m, n

def load_vars_angio( output_path, q ):

    filename_c = output_path + 'c_%i.npy' %(q)
    filename_f = output_path + 'f_%i.npy' %(q)
    filename_n = output_path + 'n_%i.npy' %(q)

    c = np.load(filename_c)
    f = np.load(filename_f)
    n = np.load(filename_n)

    return c, f, n

def rk4_angio( c, f, n, constants, cterms, fterms, nterms ):

    from plot_utils import plot_vars_angio, plot_vars_angio_noise, plot_all_terms_angio

    N = constants.N
    dt = constants.dt
    tf = constants.tf
    plot_interval = constants.plot_interval
    save_interval = constants.save_interval
    c_sum = np.sum(np.sum(c))
    f_sum = np.sum(np.sum(f))
    n_sum = np.sum(np.sum(n))
    dt_tolerance = constants.dt_tolerance

    # compute solutions:
    t = constants.t0
    q = constants.q0
    time_data = np.array([])
    while t <= tf-dt:

        print('\n*********************************************')
        print('     step = ',q)
        print('     time = %.12f s' %(t))
        print('     dt = %.12f s' %(dt))

        if round(q/plot_interval) == (round(q)/plot_interval) and q > 0:
            print('     ...saving plots')
            plot_vars_angio( c, f, n, t, q, constants )
            plot_all_terms_angio( cterms, fterms, nterms, t, q, constants )
            #plot_fterms_angio( fterms, t, q, constants )
            #plot_nterms_cterms_angio( nterms, cterms, t, q, constants )

        if round(q/save_interval) == (round(q)/save_interval) and q > 0:
            print('     ...saving data')
            save_vars_angio( c, f, n, t, q, constants )
            time_data = save_time( time_data, t, q, constants )
            save_cterms_angio( cterms, t, q, constants )
            save_fterms_angio( fterms, t, q, constants )
            save_nterms_angio( nterms, t, q, constants )

        # hapto1, hapto2 ~ grad(n) dot grad(f) , n laplacian(f)

        kc1 = cRHS_angio( c, n, constants )
        kf1, prod1, uptake1 = fRHS_angio( f, n, constants )
        kn1, diff1, chemo11, chemo12, chemo13, hapto11, hapto12 = nRHS_angio( c, f, n, constants )
        #kn1 = nRHS_angio_flux( c, f, n, constants )

        kc2 = cRHS_angio( c + dt*kc1/2., n + dt*kn1/2., constants )
        kf2, prod2, uptake2 = fRHS_angio( f + dt*kf1/2., n + dt*kn1/2., constants )
        kn2, diff2, chemo21, chemo22, chemo23, hapto21, hapto22 = nRHS_angio( c + dt*kc1/2., f + dt*kf1/2., n + dt*kn1/2., constants )
        #kn2 = nRHS_angio_flux( c + dt*kc1/2., f + dt*kf1/2., n + dt*kn1/2., constants )

        kc3 = cRHS_angio( c + dt*kc2/2., n + dt*kn2/2., constants )
        kf3, prod3, uptake3 = fRHS_angio( f + dt*kf2/2., n + dt*kn2/2., constants )
        kn3, diff3, chemo31, chemo32, chemo33, hapto31, hapto32 = nRHS_angio( c + dt*kc2/2., f + dt*kf2/2., n + dt*kn2/2., constants )
        #kn3 = nRHS_angio_flux( c + dt*kc2/2., f + dt*kf2/2., n + dt*kn2/2., constants )

        kc4 = cRHS_angio( c + dt*kc3, n + dt*kn3, constants )
        kf4, prod4, uptake4 = fRHS_angio( f + dt*kf3, n + dt*kn3, constants )
        kn4, diff4, chemo41, chemo42, chemo43, hapto41, hapto42 = nRHS_angio( c + dt*kc3, f + dt*kf3, n + dt*kn3, constants )
        #kn4 = nRHS_angio_flux( c + dt*kc3, f + dt*kf3, n + dt*kn3, constants )

        if dt_tolerance > 0.:

            dth = dt/2.

            kc2h = cRHS_angio( c + dth*kc1/2., n + dth*kn1/2., constants )
            kf2h, blank1, blank2 = fRHS_angio( f + dth*kf1/2., n + dth*kn1/2., constants )
            kn2h, blank1, blank2, blank3, blank4, blank5, blank6 = nRHS_angio( c + dth*kc1/2., f + dth*kf1/2., n + dth*kn1/2., constants )
            #kn2h = nRHS_angio_flux( c + dth*kc1/2., f + dth*kf1/2., n + dth*kn1/2., constants )

            kc3h = cRHS_angio( c + dth*kc2h/2., n + dth*kn2h/2., constants )
            kf3h, blank1, blank2 = fRHS_angio( f + dth*kf2h/2., n + dth*kn2h/2., constants )
            kn3h, blank1, blank2, blank3, blank4, blank5, blank6 = nRHS_angio( c + dth*kc2h/2., f + dth*kf2h/2., n + dth*kn2h/2., constants )
            #kn3h = nRHS_angio_flux( c + dth*kc2h/2., f + dth*kf2h/2., n + dth*kn2h/2., constants )

            kc4h = cRHS_angio( c + dth*kc3h, n + dth*kn3h, constants )
            kf4h, blank1, blank2 = fRHS_angio( f + dth*kf3h, n + dth*kn3h, constants )
            kn4h, blank1, blank2, blank3, blank4, blank5, blank6 = nRHS_angio( c + dth*kc3h, f + dth*kf3h, n + dth*kn3h, constants )
            #kn4h = nRHS_angio_flux( c + dth*kc3h, f + dth*kf3h, n + dth*kn3h, constants )

        # update continuous variables
        kf = (kf1 + 2.*kf2 + 2.*kf3 + kf4)/6.
        kc = (kc1 + 2.*kc2 + 2.*kc3 + kc4)/6.
        kn = (kn1 + 2.*kn2 + 2.*kn3 + kn4)/6.
        if dt_tolerance > 0.:
            kfh = (kf1 + 2.*kf2h + 2.*kf3h + kf4h)/6.
            kch = (kc1 + 2.*kc2h + 2.*kc3h + kc4h)/6.
            knh = (kn1 + 2.*kn2h + 2.*kn3h + kn4h)/6.
            fh = f + dth*kfh/2.
            ch = c + dth*kch/2.
            nh = n + dth*knh/2.
        f += dt*kf
        c += dt*kc
        n += dt*kn

        # no negative values:
        c,f,n = no_negative_values(c,f,n,N=constants.N)
        if dt_tolerance > 0.:
            ch,fh,nh = no_negative_values(ch,fh,nh,N=constants.N)
            dt_new_c = modify_dt( c, ch, dt, dt_tolerance )
            dt_new_f = modify_dt( f, fh, dt, dt_tolerance )
            dt_new_n = modify_dt( n, nh, dt, dt_tolerance )
            dt = min(dt_new_c,dt_new_f,dt_new_n)
            dt = max(dt,1e-9)

        # c terms:
        cterms.ddt = kc

        # f terms:
        fterms.ddt = kf
        fterms.production = (prod1 + 2.*prod2 + 2.*prod3 + prod4)/6.
        fterms.uptake = (uptake1 + 2.*uptake2 + 2.*uptake3 + uptake4)/6.

        # n terms:
        """
        ddt, diff, chemo1, chemo2, chemo3, hapto1, hapto2 = nRHS_angio( c, f, n, constants )
        nterms.ddt = kn
        nterms.diffusion = diff
        nterms.hapto1 = hapto1
        nterms.hapto2 = hapto2
        nterms.chemo1 = chemo1
        nterms.chemo2 = chemo2
        nterms.chemo3 = chemo3
        """
        # n terms:
        nterms.ddt = kn
        nterms.diffusion = (diff1 + 2.*diff2 + 2.*diff3 + diff4)/6.
        nterms.hapto1 = (hapto11 + 2.*hapto21 + 2.*hapto31 + hapto41)/6.
        nterms.hapto2 = (hapto12 + 2.*hapto22 + 2.*hapto32 + hapto42)/6.
        nterms.chemo1 = (chemo11 + 2.*chemo21 + 2.*chemo31 + chemo41)/6.
        nterms.chemo2 = (chemo12 + 2.*chemo22 + 2.*chemo32 + chemo42)/6.
        nterms.chemo3 = (chemo13 + 2.*chemo23 + 2.*chemo33 + chemo43)/6.

        print('     int(c/c0) = ',np.sum(np.sum(c))/c_sum)
        print('     max/min(c) = ',np.amax(c),np.amin(c))
        print('     int(f/f0) = ',np.sum(np.sum(f))/f_sum)
        print('     max/min(f) = ',np.amax(f),np.amin(f))
        print('     int(n/n0) = ',np.sum(np.sum(n))/n_sum)
        print('     max/min(n) = ',np.amax(n),np.amin(n))
        print('     residual(dn/dt) = ',np.amax(np.abs(nterms.diffusion+nterms.chemo1+nterms.chemo2+nterms.chemo3+nterms.hapto1+nterms.hapto2-nterms.ddt)))
        if np.isnan(np.sum(np.sum(n))):
            print('ERROR: divergent simulation')
            return c, f, n

        t = t + dt
        q = q + 1

    return c, f, n

def sinusoidal_noise( constants, Nm, scale ):
    N = constants.N # square grid resolution
    x = constants.x
    y = constants.y

    # initialize modes, etc
    #a0 = np.zeros([Nm]); b0 = np.zeros([Nm]); # amplitudes
    a = np.zeros([Nm]); b = np.zeros([Nm]); # amplitudes
    #c = np.zeros([Nm]); d = np.zeros([Nm]); # amplitudes
    pa = np.zeros([Nm]); pb = np.zeros([Nm]); # phase
    #pc = np.zeros([Nm]); pd = np.zeros([Nm]); # phase
    kxa0 = np.zeros([Nm]); kya0 = np.zeros([Nm]); # wavenumbers
    kxb0 = np.zeros([Nm]); kyb0 = np.zeros([Nm]); # wavenumbers
    kxa = np.zeros([Nm]); kya = np.zeros([Nm]); # wavenumbers
    kxb = np.zeros([Nm]); kyb = np.zeros([Nm]); # wavenumbers
    #kxc = np.zeros([Nm]); kyc = np.zeros([Nm]); # wavenumbers
    #kxd = np.zeros([Nm]); kyd = np.zeros([Nm]); # wavenumbers
    L = 10. #10. # lowest noise wavelength
    fwn = 2.*np.pi/L # fundamental wavenumber
    #if (Nm*fwn) >= (N*fwn/2.): # Nm = number of modes # Nm 2 < N
    if Nm >= N*np.pi/fwn: # Nm = number of modes # Nm 2 < N
        print('\nCAUTION: highest mode exceeds Nyquist criteria for the grid!\n Nm > %i' %(N*np.pi/fwn))
    for im in range(0,Nm):
        #kxa[im] = im*fwn*np.random.choice((-1.,1.)); kya[im] = im*fwn*np.random.choice((-1.,1.)); # 1
        #kxb[im] = im*fwn*np.random.choice((-1.,1.)); kyb[im] = im*fwn*np.random.choice((-1.,1.));
        #kxa[im] = im*fwn*np.random.normal(loc=0.0,scale=1.); kya[im] = im*fwn*np.random.normal(loc=0.0,scale=1.); # 2
        #kxb[im] = im*fwn*np.random.normal(loc=0.0,scale=1.); kyb[im] = im*fwn*np.random.normal(loc=0.0,scale=1.);
        kxa0[im] = im*fwn*np.random.uniform(-1.,1.,1); kya0[im] = im*fwn*np.random.uniform(-1.,1.,1); # cosine wavenumbers
        kxb0[im] = im*fwn*np.random.uniform(-1.,1.,1); kyb0[im] = im*fwn*np.random.uniform(-1.,1.,1); # sinusoid wavenumbers
        #kxc[im] = im*fwn*np.random.uniform(-1.,1.,1); kyc[im] = im*fwn*np.random.uniform(-1.,1.,1); # 3
        #kxd[im] = im*fwn*np.random.uniform(-1.,1.,1); kyd[im] = im*fwn*np.random.uniform(-1.,1.,1);
        #a[im] = np.random.normal(loc=0.0,scale=1.); b[im] = np.random.normal(loc=0.0,scale=1.) # 1,2,3
        #a[im] = np.random.normal(loc=0.0,scale=1.)/(2.*(im+1)); b[im] = np.random.normal(loc=0.0,scale=1.)/(2.*(im+1))
        #
        #a[im] = np.random.uniform(-1.,1.,1)/(L/2*(im+1)); b[im] = np.random.uniform(-1.,1.,1)/(L/2*(im+1))
        #
        #c[im] = np.random.uniform(-1.,1.,1)/(L/2*(im+1)); d[im] = np.random.uniform(-1.,1.,1)/(L/2*(im+1))
        #pa[im] = np.random.normal(loc=0.0,scale=2.*np.pi) # 1,2,3 phase
        #pb[im] = np.random.normal(loc=0.0,scale=2.*np.pi) # 1,2,3 phase
        #pa[im] = np.random.uniform(-np.pi,np.pi,1) # phase
        #pb[im] = np.random.uniform(-np.pi,np.pi,1) # phase
        pa[im] = (im+1.)*np.random.uniform(-2.*np.pi,2.*np.pi,1) #
        pb[im] = (im+1.)*np.random.uniform(-2.*np.pi,2.*np.pi,1) #
        #pc[im] = (im+1.)*np.random.uniform(-2.*np.pi,2.*np.pi,1) #
        #pd[im] = (im+1.)*np.random.uniform(-2.*np.pi,2.*np.pi,1) #

    # shuffle wavenumbers so that the wave vector points in different directions
    #np.random.shuffle(kxa); np.random.shuffle(kya)
    #np.random.shuffle(kxb); np.random.shuffle(kyb)
    kx_locs = np.random.randint(0, high=Nm, size=Nm, dtype=int)
    ky_locs = np.random.randint(0, high=Nm, size=Nm, dtype=int)
    for im in range(0,Nm):
        kxa[im] = kxa0[kx_locs[im]]
        kxb[im] = kxb0[kx_locs[im]]
        kya[im] = kya0[ky_locs[im]]
        kyb[im] = kyb0[ky_locs[im]]
        a[im] = np.random.uniform(-1.,1.,1)/((L**2.)*np.sqrt((kx_locs[im]+1)**2.+(ky_locs[im]+1)**2.)) # 11/11/20
        b[im] = np.random.uniform(-1.,1.,1)/((L**2.)*np.sqrt((kx_locs[im]+1)**2.+(ky_locs[im]+1)**2.))
        #a[im] = np.random.uniform(-1.,1.,1)/(L*np.sqrt((kx_locs[im]+1)**2.+(ky_locs[im]+1)**2.)) # before 11/11/20
        #b[im] = np.random.uniform(-1.,1.,1)/(L*np.sqrt((kx_locs[im]+1)**2.+(ky_locs[im]+1)**2.))
        #a[im] = a[im]/(L/2*(kx_locs[im]+1)+L/2*(im+1))
        #b[im] = b0[kxb_locs[im]]

    # make noise!
    noise = np.zeros([N,N])
    #xc = 0.5 # x coordinate, tumor center
    #yc = 0.5 # y coordinate, tumor center
    for i in range(0,N):
        for j in range(0,N):
            #r = np.sqrt( (x[i]-xc)**2.+(y[j]-yc)**2. )
            for im in range(0,Nm): # summing over modes
                #noise[j,i] += a[im]*np.cos(kxa[im]*r+pa[im])+b[im]*np.sin(kxb[im]*r+pb[im])
                noise[j,i] += a[im]*np.cos(kxa[im]*x[i]+kya[im]*y[j]+pa[im])+b[im]*np.sin(kxb[im]*x[i]+kyb[im]*y[j]+pb[im])
    noise = (noise-np.mean(noise)) # centers at zero
    noise = noise/np.amax(np.abs(noise))*scale # scales
    #noise = noise/np.std(noise)*scale

    return noise

def smoothed_random_noise( constants , var_name , noise_type ):
    N = constants.N
    if var_name == 'c':
        amp = constants.noise_amp_c
        nu_dt = constants.noise_nu_dt_c
        iter = constants.noise_iterations_c
    elif var_name == 'f':
        amp = constants.noise_amp_f
        nu_dt = constants.noise_nu_dt_f
        iter = constants.noise_iterations_f
    elif var_name == 'm':
        amp = constants.noise_amp_m
        nu_dt = constants.noise_nu_dt_m
        iter = constants.noise_iterations_m
    elif var_name == 'n':
        amp = constants.noise_amp_n
        nu_dt = constants.noise_nu_dt_n
        iter = constants.noise_iterations_n

    noise = np.zeros([N,N])
    if amp == 0.:
        return noise
    else:
        for i in range(0,N):
            for j in range(0,N):
                if noise_type == 'Gaussian':
                    noise[j,i] = np.random.normal(loc=0.0,scale=amp)
                elif noise_type == 'uniform':
                    #noise[j,i] = np.random.uniform(-amp,amp,1)
                    noise[j,i] = np.random.uniform(0.,amp,1)
        if iter > 0:
            for k in range(0,iter):
                noiseg = get_ghosts( noise, constants )
                noise += nu_dt*laplacian( noiseg, constants )
            return noise
        elif iter == 0:
            return noise


def modify_dt( var, varh, dt, dt_tolerance ):
    # Kendall E. Atkinson, Numerical Analysis, Second Edition, John Wiley & Sons, 1989
    max_lte = np.amax(np.abs(var-varh))
    err = np.sqrt(dt_tolerance*0.5*np.power(max_lte,-1.))
    dt_new = 0.9*dt*min( max(err,0.3) , 2.0 )
    return dt_new

def get_restart_time( q , output_path ):
    filename = output_path + 't_%i.npy' %(q)
    time_series = np.load(filename)
    Nt = len(time_series)
    t_restart = time_series[Nt-1]
    return t_restart

def no_negative_values( *args, N ):
    for k in range(0,np.shape(args)[0]):
        var = args[k]
        for i in range(0,N):
            for j in range(0,N):
                if var[i,j] <= 0.0:
                   var[i,j] = 0.0
    return args

def upper_limit( var, N, threshold ):
    # this function is equivalent to adding at term to the temporal
    # evolution d/dt(arg) = - 1/dt*(arg-arg0)*delta(arg-arg0), where delta
    # is the Dirac delta function.
    # In other words, it relaxes the value of arg back to arg0 if arg > arg0,
    # where arg0 is representative of a maximum concentration or density at
    # a grid cell.
    #for k in range(0,np.shape(args)[0]):
    #    var = args[k]
    for i in range(0,N):
        for j in range(0,N):
            if var[i,j] > threshold:
                var[i,j] = threshold
    return var
