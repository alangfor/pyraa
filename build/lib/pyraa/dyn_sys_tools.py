"""
    PyDAAT
    ________________________________________________
    Dynamical System Tools

    Methods for Dynamical Systems Analysis

    Written in: Python 3
    Author: Drew Langford
    Last Edited: 8/2/21


    List of Functions
    ------------------------------------------------

    STMs
    henon_curuve
    x_zvc_func
    FTLE
    JCs
    poincare
    manifold_state
    shooter_update_state
    multi_shooter_update_state
    init_shooter
    init_patches
    eval_dxdtau
"""

from array import array
import numpy as np
from pyraa.models import Models
import numba as nb

@nb.njit()
def calc_RV2OE(states, verbose = False):
    """
        calc_OE - function which takes state vector, S
            and calculates the six orbital elements

            f - true anamoly [deg]
            a - semi-major axis [km]
            e - eccentricity [0, 1)
            i - inclination [deg]
            Om - RA of ascending node [deg]
            w - argument of perigee [deg]  

            Args:
                S (1x6 array): state vector [km]/[km/s]
                    x, y, z, vx, vy, vz in ECI frame  
                
                Opt Args:
                verbose (bool): if True, prints solved OE

            Returns:
                OE_array (1x6 array) - solved orbital elements
                    note: all angle quants in deg
                    [f, a, e, i, Om, w]
    
    """

    N = len(states)
    OE_states = np.zeros((N, 6))

    for index in range(N):
        r_ = states[index, 0:3]
        rx, ry, rz = r_
        v_ = states[index, 3:]
        vx, vy, vz = v_
        r = np.linalg.norm(r_)
        v = np.linalg.norm(v_)
        vr = np.dot(v_, r_/r)

        # Angular momentum 
        # Technically OE but not used here
        h_ = np.cross(r_, v_)
        h = np.linalg.norm(h_)
        hz = h_[2]

        N_ = np.array([1.0, 0.0, 0.0])
        Nx, Ny, Nz = N_
        N = np.linalg.norm(N_)

        a = 1/(2/r - v**2)
        e = np.sqrt(1 - h**2/a)
        i = np.arccos(hz/h)

        # RA of Ascending Node 
        if Ny >= 0.0:
            Om = np.arccos(Nx/N)
        else:
            Om = 2*np.pi - np.arccos(Nx/N)

        # # Node line calc | Note quad ambiguity
        # K_ = np.array([0, 0, 1])
        # N_ = np.cross(K_, h_)
        # N = np.linalg.norm(N_)
        # Nx, Ny, Nz = N_

        ### Begin OE Calculations

        # Eccentrcity calc

        e_ = np.cross(v_, h_) - r_/r
        e = np.linalg.norm(e_)
        ex, ey, ez = e_

        # Semi-major axis calc
        a = h**2/(1-e**2)

        # Inclination calc
        i = np.arccos(h_[2]/h)

        if ey >= 0:
            w = np.arccos(np.dot(N_/N, e_/e))

        else:
            w = 2*np.pi - np.arccos(np.dot(N_/N, e_/e))

        if vr >= 0:
            f = np.arccos(np.dot(e_/e, r_/r))
        else:
            f = -np.arccos(np.dot(e_/e, r_/r))

        i_deg = np.rad2deg(i)
        Om_deg = np.rad2deg(Om)
        w_deg = np.rad2deg(w)
        f_deg = np.rad2deg(f)

        OE_states[index] = np.array([f_deg, a, e, i_deg, Om_deg, w_deg])

    return OE_states


@nb.njit(parallel = True, debug = False)
def transform_mult(Qs, vecs, N):

    transform_vec = np.zeros((N, 6), dtype = np.double)
    for i in nb.prange(N):
        transform_vec[i] = Qs[i] @ vecs[i]

    return transform_vec


def STMs(sc):
    """ STMs - Calculates and assigns the STMs to a sc

            Args: 
                sc (obj): spacecraft object 

            Returns:
                None
    """

    states = sc.get_states().T
    Nstates = len(states)
    STMs = np.ndarray((Nstates, 6, 6))

    for i in range(Nstates-1):
        Xi = states[i]
        Xip1 = states[i+1]
        STMs[i] = np.outer(Xip1, Xi)/(np.linalg.norm(Xi))**2

    STMs = np.reshape(STMs, (36, Nstates))
    sc.set_STM(STMs)

    return None

def henon_curve(x, JC, mu):
    """ henon_curve - calculates boundary curve of Henon section [y = 0] Poinacre map

            This function uses the integral constant to compute xdot, when y,ydot=0 for a given x

            Args:
                x (1-D array) : array of x values to compute 
                JC (float) : Chosen CR3BP Jacobi integral constant
                mu (float) : mass ratio of primaries, m2/(m1 + m2)

            Returns 
                x, xdot (tuple) : masked x and xdot values of the range x.min to x.max which 
                    a satisfy the integral constant relation
    
    """

    xdotsq = x**2 +  2*( (1-mu)/np.abs(x + mu) + mu/np.abs(x-(1-mu)) ) - JC
    
    mask = xdotsq > 0
    xdot = np.sqrt( xdotsq[mask] ) 
    
    return x[mask], xdot

def x_zvc_func(x, mu, JC, right = False, left = False):
    """ x_zvc_func - returns the zero velocity curve functions 
        with zero crossings correpsonding to the zvc boundary

            Args:
                x (1-D array) : array of x values to compute the function
                mu (float) : mass ratio of primaries, m2/(m1 + m2)
                JC (float) : Chosen CR3BP Jacobi integral constant

            Returns:
                F (1-D array) : array of zvc func values
    
    """
    
    sign = +1
    if right == True:
        sign = -1
        
    sign2 = +1
    if left == True:
        sign2 = -1
    
    F = JC - ( x**2 + 2*( -sign*(1-mu)/(x+mu) -sign2* mu/(x-(1-mu)) ) )
    
    return F

def FTLE(sc, dynamics):
    """ FTLE - calculates the Finite Time Lyapunov Exponents 
        over a spacecraft trajectory and adds it to the sc properties

            Args:
                sc (object): propogated spacecraft object 
                dynamics (str): dynamics of the sim

    """

    # Retrieve sc info
    taus = sc.get_taus()
    STMs = sc.get_STMs()
    states = sc.get_states()

    # Initialize an array for FTLEs
    FTLEs = np.zeros_like(taus)

    # Calculate FTLEs over trajectory
    model = Models(dynamics)
    jacobian = model.get_jacobian_func()
    for i, tau in enumerate(taus[:-1]):

        # Can't calc STM for tf
        if i != len(taus) - 1:
            DT = np.abs(taus[i+1]-taus[i])

        # STM and state vector
        Phi = STMs[i]
        s = states.T[i]
        dPhi = jacobian(s, tau) @ Phi

        # C-G Tensor?? Also could be Phi.T @ Phi.. Literature uses both
        #M = dPhi.T @ dPhi
        M = Phi.T @ Phi

        sqrtM = np.lib.scimath.sqrt(M)
        eigvals = np.linalg.norm(np.linalg.eigvals(sqrtM))
        lmax = np.max(eigvals)

        FTLEs[i] = np.log(lmax) #*(np.exp(-np.power(tau, 31/64)))

    sc.set_FTLEs(FTLEs)

def JCs(sc, model, mu, e):
    """ JCs - calculates Jacobi Constant 
        over a spacecraft trajectory and adds it to the sc properties

            Args:
                sc (object): propogated spacecraft object 
                dynamics (str): dynamics of the sim

    """

    # Retrieve sc info
    taus = sc.get_taus()
    states = sc.get_states()

    # Initialize an array for FTLEs
    JCs = np.zeros_like(taus)
    JC_func = model.get_JC_func()
    for i, tau in enumerate(taus):
        s = states.T[i]
        JCs[i] = JC_func(tau, s, mu, e)

    #sc.set_JCs(JCs)
    return JCs

def poincare(tau, state, eval_STM, pevent, epoch):
    """ poincare - poincare 'event' function for use in 
        simulation.propogate() func.

            Args:
                tau, state, eval_STM, epoch: necessary arguments for
                 sci.integrate.solve_ivp
                pevent (tuple): 
                    n: state element to evaluate
                    t: trigger value to record event

                Example:
                    I want to record x-z plane crossings. Therefore, I need
                    to record when the y component is zero.

                    n : y is the 1 index in the state vector
                    t : the trigger value should be set to 0

                    pevent = [1, 0]
    
    """
    
    n = pevent[0]
    t = pevent[1]

    f = state[n] - t

    return f

def manifold_state(sc):
    """ manifold_state - produces unstable and stable initial states to 
        generate orbit manifolds - requires sc trajectory to be precisely periodic

        Manifolds are topological surfaces which define the natural path trajectories
        in and out of a precisiely periodic orbit. Stable manifolds are the paths into
        the orbit. Unstable manifolds are the paths out of the orbit. 

            Args:
                sc (object): spacecraft object holding precisely one period
                    orbit trajectory

            Returns:
                um_states (array): unstable manifold IC, propogate forward
                    to generate unstable manifold
                
                sm_states (array): stable manifold IC, propogate backward
                    to generate stable manifold.
    
    """

    STMs = sc.get_STMs()
    states = sc.get_states().T

    s0 = states[-1]
    M = STMs[-1]
    det = np.linalg.det(M)

    print(det)

    eigvals, eigvecs = np.linalg.eig(M)
    val_pairs = np.reshape(eigvals, (3, 2))
    vec_pairs = np.reshape(eigvecs.T, (3, 2, 6))

    um_states = np.empty(6)
    sm_states = np.empty(6)

    print()
    print('Monodromy Matrix Eigenspectra')
    print('_'*50)
    for i, val_pair in enumerate(val_pairs):
        imag = np.imag(val_pair[0])
        if imag != 0:
            print('Oscillatory Pair')
            print('e1: {:5.5} | e2: {:5.5}'.format(val_pair[0], val_pair[1]))
            print()
            continue
        elif np.abs(np.real(val_pair[0]) - 1) < 1e-2:
            print('Periodic Pair')
            print('e1: {:5.5} | e2: {:5.5}'.format(val_pair[0], val_pair[1]))
            print()
            continue
        else:
            print('Exponential Pair')
            print('e1: {:5.5} | e2: {:5.5}'.format(val_pair[0], val_pair[1]))
            print()

            e_us = []
            e_ss = []     
            for j, l in enumerate(val_pairs[i]):
                norm = np.linalg.norm(l)
                if norm > 1:
                    e_u = vec_pairs[i, j]
                    e_us.append(e_u)
                    print('Unstable Eigendirection')
                    print(e_u)
                elif norm < 1:
                    e_s = vec_pairs[i, j]
                    e_ss.append(e_s)
                    print('Stable Eigendirection')
                    print(e_s)
        print()

    d = 1e-4
    A = np.linalg.norm(s0[0:3])
    for e_u in e_us:
        um_state1 = np.real(s0 + d*(e_u/A))
        um_state2 = np.real(s0 - d*(e_u/A))
        um_states = np.vstack((um_states, um_state1, um_state2))

    for e_s in e_ss:
        sm_state = np.real(s0 + d*e_s/A)
        sm_states = np.vstack((sm_states, sm_state))


    return um_states, sm_states

def shooter_update_state(sc, X_els, FX_els, X_d, Td, dynamics, mu, e, s, Js):
    """ shooter_update_state - calculates updated design variables for 
        single shooting method.
        For use in simulation.Simulation.single_shooter method

            Args:
                X_els (1xN array): state array index of design variables
                FX_els (1xN array): state arrat index of constraint variables
                X_d (1xN): constraint variable desired values
                Td (flt): time constraint value

                dynamics (str): dynamics of the sim

            Returns:
                X_ip1 (1xN array): updated design variable vector
                FX_i (1xN array): constraint vector 

    """

    model = Models(dynamics)
    state_func = model.get_eom_func()

    # Retrieve propogated trajectory info
    S_0 = sc.get_states().T[0]
    tau0 = sc.get_taus()[0]
    S_f = sc.get_state()
    tauf = sc.get_tau()
    STMf = sc.get_STMs()[-1]
    dS_f = state_func(tauf, S_f, mu, e, s, Js, eval_STM = False)

    # Create the design variable array, X
    n = len(X_els)
    X_i = np.zeros(n)
    for el, X_el in enumerate(X_els):
        if X_el == 6:
            X_i[el] = tauf
        else:
            X_i[el] = S_0[X_el]

    # Create the constraint variable array, F(X)
    c = len(FX_els)
    FX_i = np.zeros(c)
    for el, FX_el in enumerate(FX_els):
        if FX_el == 6:
            FX_i[el] = tauf - Td 
        else:
            FX_i[el] = S_f[FX_el] - X_d[el]

    # Create the Jacobian matrix
    DFX_i = np.zeros((c, n))
    for i, row in enumerate(DFX_i):
        for j, col in enumerate(DFX_i[0]):
            x_t = FX_els[i]
            x_0 = X_els[j]
            if x_0 != 6:
                DFX_i[i, j] = STMf[x_t, x_0]
            else:
                DFX_i[i, j] = dS_f[x_t]


    # Calculate updated design variable array
    X_ip1 = X_i - DFX_i.T @ np.linalg.inv(DFX_i @ DFX_i.T) @ FX_i

    return X_ip1, FX_i, DFX_i

def multi_shooter_update_state(sim, Td, patches, epochs, TOFs, N, dynamics, ax = None):
    """ multi_shooter_update_state - calculates updated design and constraint 
        vectors for the multiple shooting method.
        For use in the simulation.Simulation.multi_shooter method

            Args:
                sim (class instance): Simulation instance
                Td (flt): desired final time
                patches (Nx6 array): patch points
                epochs (1xN array): epochs of patch points
                N (float): number of patch points
                dynamics (str): dynamics of the sim

            Returns:
            X_ip1 (1xN array): updated design variable vector
            FX_i (1xN array): constraint vector 

    """

    # State vectors for each arc
    dS_fs = np.zeros([N, 6]) # derivative of final states
    S_fs = np.zeros([N, 6]) # final state vectors
    S_0s = np.zeros([N, 6]) # inititial state vectors
    STMs = np.zeros([N, 6, 6]) # final STMs
    dxdtaus = np.zeros([N, 6]) # state with respect to epoch
    tau_fs = np.zeros(N) # final epochs

    calc_epoch = True
    
    # Propogates each patch point and fills state vectors^^
    for i, s in enumerate(patches):

        # Get patch's initial epoch
        tau_0 = epochs[i]
        
        # Create sc from patch point and epoch
        sc = sim.create_sc(s, tau_0, verbose = False)

        # Propogate the 1 sc for specified TOF
        sim.propogate(TOFs[i], tol= [1e-12, 1e-10], epoch = False, verbose = False)

        tauf = sc.get_tau()
        S_f = sc.get_state()
        
        # Fill state vectors from propogated sc
        tau_fs[i] = tauf
        dS_fs[i, :] = sim.state_func(tauf, S_f, eval_STM = False)
        S_fs[i, :] = S_f
        S_0s[i, :] = sc.get_states().T[0]
        STMs[i] = sc.get_STM()
        dxdtaus[i] = eval_dxdtau(S_f, tau_0, sim)

        if ax != None: sim.plot_orbit(ax = ax)
        sim.clear_scs()
        
    # Create the constraint function
    FX_i = np.zeros((N, 6))
    for p_num in range(len(FX_i)): # State continuity constraint
        if p_num == len(FX_i) - 1:
            dX = S_fs[p_num] - S_0s[0]
        else:
            dX = S_fs[p_num] - S_0s[p_num+1]
        FX_i[p_num, :] = dX

    FX_iT = np.sum(TOFs) - Td # Total TOF constraint

    FX_itau = np.zeros(N-1)
    for p_num in range(len(FX_itau)): # epoch continuity constraint
        FX_itau[p_num] = epochs[p_num] + TOFs[p_num] - epochs[p_num+1]
    
    # Constraint function of middle pathces, final patch, and epoch continuities
    FX_i = np.concatenate((FX_i.flatten(), [S_0s[0, 1]],  FX_itau, [FX_iT]))

    # Create the design function
    X_i = np.zeros((N, 6))
    for p_num in range(len(X_i)):
        X_i[p_num, :] = S_0s[p_num]

    X_i = np.concatenate((X_i.flatten(), epochs, TOFs)) # design vector

    # Generate Jacbian -- set for 4 patch points
    row1 = np.hstack([STMs[0]        , -np.identity(6), np.zeros((6,6)), np.zeros((6,6)), dxdtaus[0][:, None], np.zeros(6)[:, None], np.zeros(6)[:, None], np.zeros(6)[:, None], dS_fs[0][:, None], np.zeros(6)[:, None], np.zeros(6)[:, None], np.zeros(6)[:, None]])
    row2 = np.hstack([np.zeros((6,6)), STMs[1]        , -np.identity(6), np.zeros((6,6)), np.zeros(6)[:, None], dxdtaus[1][:, None], np.zeros(6)[:, None], np.zeros(6)[:, None], np.zeros(6)[:, None], dS_fs[1][:, None], np.zeros(6)[:, None], np.zeros(6)[:, None]])
    row3 = np.hstack([np.zeros((6,6)), np.zeros((6,6)), STMs[2]        , -np.identity(6), np.zeros(6)[:, None], np.zeros(6)[:, None], dxdtaus[2][:, None], np.zeros(6)[:, None], np.zeros(6)[:, None], np.zeros(6)[:, None], dS_fs[2][:, None], np.zeros(6)[:, None]])
    row4 = np.hstack([-np.identity(6), np.zeros((6,6)), np.zeros((6,6)), STMs[3]        , np.zeros(6)[:, None], np.zeros(6)[:, None], np.zeros(6)[:, None], dxdtaus[3][:, None], np.zeros(6)[:, None], np.zeros(6)[:, None], np.zeros(6)[:, None], dS_fs[3][:, None]])
    row5 = np.hstack([[0,1,0,0,0,0]  , np.zeros(6), np.zeros(6), np.zeros(6), 0, 0, 0, 0, 0, 0, 0, 0])
    row6 = np.hstack([np.zeros(6)    , np.zeros(6), np.zeros(6), np.zeros(6), 1, -1, 0, 0, 1, 0, 0, 0])
    row7 = np.hstack([np.zeros(6)    , np.zeros(6), np.zeros(6), np.zeros(6), 0, 1, -1, 0, 0, 1, 0, 0])
    row8 = np.hstack([np.zeros(6)    , np.zeros(6), np.zeros(6), np.zeros(6), 0, 0, 1, -1, 0, 0, 1, 0])
    row9 = np.hstack([np.zeros(6)    , np.zeros(6), np.zeros(6), np.zeros(6),  0, 0, 0, 0, 1, 1, 1, 1])


    DFX_i = np.vstack([row1, row2, row3, row4, row5, row6, row7, row8, row9])
    
    ###### Print Jacobian ######
    # print('\n'.join([''.join(['{:10.5}'.format(item) for item in row]) 
    #     for row in DFX_i]))

    ##### Print vector shapes #####
    # print(np.shape(X_i))
    # print(np.shape(FX_i))
    # print(np.shape(DFX_i))

    # Calculate updated design variables
    X_ip1 = X_i - (DFX_i.T @ np.linalg.inv(DFX_i @ DFX_i.T) @ FX_i)*0.7

    return X_ip1, FX_i

def init_shooter(s_b, T_b, Xd_b):
    """ init_shooter - generates arrays of design and contraint
        variable indices

        For use in simulation.Simultion.single_shooter

            Args:
                s_b (1x6 array): state boolean array - sets design variables
                T_b (1x1 array): time boolean array - sets time as design variable
                Xd_b (1x6): sets which state elements are the constraints

            Returns:
                X_els (1xN array): array of design element indices
                FX_els (1xN array): array of constraint element indicies
    """

    n = sum(s_b) + sum(T_b)
    X_els = []
    for i, s_bool in enumerate(s_b):
        if s_bool:
            X_els.append(i)
    if T_b[0]:
        X_els.append(6)

    FX_els = []
    for i, Xd_bool in enumerate(Xd_b):
        if Xd_bool:
            FX_els.append(i)

    return X_els, FX_els

def init_patches(sc, N):
    """ init_patchs - generate equal temporal patch points along 
        a sc trajectory

        For use in simulation.Simulation.multi_shooter

        Args:
            sc (sc object): propogated spacecraft 
            N (int): # of patchpoints

        Return:
            patches (Nx6): equaly spaced patch points
            TOFs (Nx1): Time of flights between patch points
    
    """

    taus = sc.get_taus()
    states = sc.get_states().T

    # Create array of patch states and their TOFs
    patches = np.zeros([N, 6])
    TOFs = np.zeros(N)

    pts = np.linspace(0, len(taus)-1, N+1)
    for i, pt in enumerate(pts[:-1]):
        patches[i, :] = states[int(pt)]
        TOFs[i] = taus[int(pts[i+1])] - taus[int(pts[i])]

    return patches, TOFs

def eval_dxdtau(s, tau0, sim):
    """ eval_dxdtau - evaluates the partial state derivative w/
        respect to epoch-- needed for epoch continuity in shooting
        algorithms

            Args:
                s (6x1 array): state at eval time
                tau0 (flt): epoch at eval time
                sim (class object): current sim instance

            returns:
                dxdtau (1x6 array): derivative of state w respect to
                    epoch
    
    """

    dtau = 1e-7

    # Clear sc and propogate for small time
    sim.clear_scs()
    #sim.create_sc(s, tau_0 = tau0, verbose = False)
    sim.create_sc(s, tau_0 = tau0 - dtau, verbose = False)
    sim.create_sc(s, tau_0 = tau0 + dtau, verbose = False)
    sim.propogate(1e-2, epoch = False, verbose = False)

    # Compute modified Euler for 3rd order f'(x) approximation
    x0 = np.double(sim.scs[0].get_state())
    x1 = np.double(sim.scs[1].get_state())
    dxdtau = (x1 - x0)/(2*dtau)
    sim.clear_scs()

    # Recreate sc object at tau0
    sim.create_sc(s, tau0)

    return dxdtau



class timer(object):
    """
    timer - basic timer class for performance testing 

    Args:
        None

    Returns:
        None
    """
    def __init__(self):
        
        pass
        
    def start(self,):
        """
        start - starts timer
        """
        
        self.t0 = time.perf_counter()
        
    def stop(self, verbose = False):
        """
        stop - stops timer

        Args:
            verbose (bool, optional): if True, prints self.dt
        """
        
        self.tf = time.perf_counter()
        
        self.dt = self.tf - self.t0
        
        if verbose:
            self.print_dt()
        
        return self.dt

    def print_dt(self,):
        """
        print_dt - prints self.dt with format
        
        """

        print('DT {} [s] \n'.format(self.dt))



if __name__ == "__main__":

    T_DRO = 3.04253432446400
    DRO = np.array([1.17, 0, 0,0, -0.489780292125578, 0])

    s_bools = [True, False, False, False, True, False]
    T_bool = [True]

    init_shooter(DRO, T_DRO, s_bools, T_bool)
