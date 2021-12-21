"""
    High-fIdelity multi-dody Dynamics Explorer (HIDE)
    _________________________________________________
    Simulation.py

    Author: Drew Langford

"""

# External packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate
import multiprocessing as mp
import functools as ft
import time
import os
import imageio
from datetime import date
from datetime import datetime
import progressbar

# HIDE modules
from models import Models
from spacecraft import Spacecraft
import dyn_sys_tools as tools

import h5py
import pathlib

class Simulation:

    def __init__(self, dynamics, verbose = True):
        """
            Simulation class:
                Initialize a simulations enviroment with specified dynamics

                    Args:
                        dynamics (str): dynamical model of the simulated enviroment
                            Options: ['CR3BP', 'ER3BP', 'BCR4BP', 'BER4BP']

                    Returns:
                        None
        """

        # Import dynamics
        self.dynamics = dynamics
        self.model = Models(dynamics)
        
        self.state_func = self.model.get_state_func()
        self.transform = self.model.get_transform_func()
        self.mu, self.l_st, self.t_st = self.model.get_char_vals()[0:3]
        pstates_func = self.model.get_pstates_func()

        # State of the earth and moon... this may change for other models?
        self.e_state_syn, self.m_state_syn  = pstates_func(0)

        # Initialize an empty list of sc objects
        self.scs = []

        self.axis_dict = {
            'x' : 0,
            'y' : 1,
            'z' : 2,
            'xdot' : 3, 
            'ydot' : 4,
            'zdot' : 5
        }

        if verbose:
            # Welcome message
            print()
            print('Welcome to the High-Fidelity Multi-Body Dyanamics Explorer!')
            print('-'*50)
            print()
            print('Simulation with {} dynamics initialized..'.format(dynamics))
            print('-'*50)
            print('-'*50)
            print()

    def create_sc(self, s0, tau_0 = 0, monte = False, verbose = False):
        """
            create_sc:
                Creates new spacecraft object with initial state

                    Args:
                        s0 (1x6 array): initial non-dim state vector of sc
                        tau_0: initial non-dim time of sc

                Returns:
                    sc (class inst): created sc object
        """
        # Initialize a spacecraft object
        sc = Spacecraft(s0, tau_0, monte = monte)
        
        # Add to scs list
        self.scs.append(sc)

        if verbose == True:
            print('New spacecraft created..')
            print('Total spacecraft: {}'.format(len(self.scs)))
            print()

        return sc

    def load_sc(self, states = None, taus = None):

        s0 = np.array([0,0,0,0,0,0])
        sc = self.create_sc(s0)

        # sc.set_state(states)
        # sc.set_tau(taus)

        sc.states = states
        sc.taus = taus



    def ydotconst_func(self, tau, x, jc):

        # x, y, z, xdot, ydot, zdot = s
        # e_state, m_state = self.model.cr3bp_pstates(tau)

        # r13 = np.linalg.norm(s[0:3] - e_state[0:3])
        # r23 = np.linalg.norm(s[0:3] - m_state[0:3])

        Ust = 0.5*(x**2) + (1-self.mu)/np.sqrt((x + self.mu)**2)+ self.mu/np.sqrt((x - (1-self.mu))**2)
        
        ydot = np.sqrt(2*Ust - jc)

        return ydot

    def G(self, x, JC):
        
        Ust = 0.5*(x**2) + (1-self.mu)/np.abs(x + self.mu) + self.mu/np.abs(x - (1-self.mu))
        
        G = np.sqrt(2*Ust - JC)
        
        return G

    def xy(self, x, JC, N):
        
        k = np.arange(0, N, 1)

        G = self.G(x, JC)
        
        #5*np.pi/6*k/N + np.pi/12
        xdot = G*np.cos(5*np.pi/6*k/N + np.pi/12)
        ydot = G*np.sin(5*np.pi/6*k/N + np.pi/12)

        Xdot = np.concatenate((xdot, xdot))
        Ydot = np.concatenate((ydot, -ydot))
        
        return Xdot, Ydot

    def create_poincare_sc(self, JC, LMR, lim = 1.5, tau_0 = 0, density = 10, 
    plot_ICs = True, monte = False, verbose = False):
        """
            create_poincare_sc - efficiency based Poincaré map initial conditions

                Args:
                    JC (float): Poincaré map Jacobi Constant
                    LMR (4x1 list): Number of 'Left, Middle, Right' ICs to produce
                        L - Left of the ZVC
                        M1 - Between the ZVC and L1 point
                        M2 - Between the L1 point and ZVC
                        L2 - Right of the ZVC
                        
                Opt. Args:
                    tau_0 (float):     
        """

        self.JC = JC 
        self.LMR = LMR 
        self.density = density
        self.tau_0 = tau_0

        # Think about adding 
        L, M1, M2, R = LMR
        xs_left = np.linspace(-lim, -1.05*(self.mu), 1000)
        xs_middle = np.linspace(-self.mu*(0.05), (1-self.mu)*0.95, 1000)
        xs_right = np.linspace((1-self.mu)*1.05, lim, 1000)

        fx_left = tools.x_zvc_func(xs_left, self.mu, JC, right = False)
        fx_middle = tools.x_zvc_func(xs_middle, self.mu, JC, right = True)
        fx_right = tools.x_zvc_func(xs_right, self.mu, JC, right = True, far = True)

        # Past L3
        mask_L = (fx_left < 0) & (xs_left < xs_left[np.argmax(fx_left)])
        # Around M1
        mask_LM = (fx_left < 0) & (xs_left > xs_left[np.argmax(fx_left)])
        mask_ML = (fx_middle < 0) & (xs_middle < xs_middle[np.argmax(fx_middle)])
        # Around M2
        mask_MR = (fx_middle < 0) & (xs_middle > xs_middle[np.argmax(fx_middle)])
        mask_RM = (fx_right < 0) & (xs_right < xs_right[np.argmax(fx_right)])
        # Past L2
        mask_R = (fx_right < 0) & (xs_right > xs_right[np.argmax(fx_right)])

        ### THIS IS THE RIGHT FORMAT
        sp_1 = int(len(xs_left[mask_L])/L)
        xs_L = xs_left[mask_L][0::sp_1]

        sp_1 = int(np.floor(len(xs_left[mask_LM])/(M1/2)))
        sp_2 = int(np.floor(len(xs_middle[mask_ML])/(M1/2)))
        xs_M1 = np.concatenate(
            (xs_left[mask_LM][0::sp_1], 
            xs_middle[mask_ML][0::sp_2]
            ))
        sp_1 = int(np.floor(len(xs_middle[mask_MR])/(M2/2)))
        sp_2 = int(np.floor(len(xs_right[mask_RM])/(M2/2)))
        xs_M2 = np.concatenate(
            (xs_middle[mask_MR][0::sp_1], 
            xs_right[mask_RM][0::sp_2]
            ))
        sp_1 = int(len(xs_right[mask_R])/R)
        xs_R = xs_right[mask_R][0::sp_1]

        #raise Exception('Testing')
        
        Xs = np.concatenate((xs_L, xs_M1, xs_M2, xs_R))
        #Xs = np.concatenate((xs_left[mask_left], xs_middle[mask_middle], xs_right[mask_right]))

        N = len(Xs)
        D = 2*density
        xdots = np.ndarray(D*N)
        ydots = np.ndarray(D*N)
        xs = np.ndarray(D*N)
        for n in range(N):
            xdot, ydot = self.xy(Xs[n], JC, density)
            xdots[n*D:(n+1)*D] = xdot
            ydots[n*D:(n+1)*D] = ydot
            xs[n*D:(n+1)*D] = Xs[n]

        Odm = np.zeros_like(xs)
        P_ICs = np.concatenate((xs[:,None].T, Odm[:,None].T, 
            Odm[:,None].T, xdots[:,None].T, ydots[:,None].T, Odm[:,None].T)).T
        for state in P_ICs:
            if verbose:
                print('New Poincaré spacecraft created..')
            self.create_sc(state, tau_0, monte = True)
        print('{} Poincaré sc created..'.format(len(P_ICs)))

        if plot_ICs:
            plt.plot(P_ICs[:, 0], P_ICs[:, 4], ls = 'none', marker = '.', ms = 10)
            self.plot_henon(JC)
            plt.xlim(np.min(P_ICs[:, 0])-1, np.max(P_ICs[:, 0])+1)
            plt.ylim(np.min(P_ICs[:, 4])-1, np.max(P_ICs[:, 4])+1)

        return P_ICs

    def create_monte_sc(self, s, tau_0 = 0, N = 10, 
        ds = [10e-3, 10e-3, 10e-3, 1e-3, 1e-3, 1e-3]):
        """
            create_monte_sc:
                Initializes sc based on a randomly peturbed model state

                    Args:
                        s0 (1x6 array): model non-dim state
                        tau_0 = initial non-dim time of mc_scs
                        N = # of monte sc to create
                    
                    Opt Args:
                        ds (1x6 array): array of perturb ([km] and [km/s])
                            dX, dY, dZ, dXdot, dYdot, dZdot

                    Returns:
                        None
        """
        ds = np.array(ds)
        # Creates array of peturbations
        ds_pos = (ds[0:3][:, None] * np.random.randn(3, N))/self.l_st
        ds_vel = (ds[3:][:, None] * np.random.randn(3, N))*(self.t_st/self.l_st)
        ds = np.concatenate((ds_pos, ds_vel))

        # Add perturbations to model state
        states = (ds + s[:, None]).T

        # Initialize new monte sc
        for state in states:
            print('New monte spacecraft created..')
            self.create_sc(state, tau_0, monte = True)

    def create_mainifold_sc(self, p_sc, T, N, stable = False):
        """
            create_manifold_sc:
                creates N spacecraft equally spaced around perodic orbit, which
                    correspond to orbit manifold ICs #Uncomment end of function

                ** this method uses dyn_sys_tools.maninfold_state()
                    Not proven to totally work, may need further development

                ** p_sc must be a very precise periodic IC, none of that literature IC shit

                Args: 
                    p_sc (sc object): a periodic IC 
                    T (float): period of the IC's orbit
                    N (int): number of orbit mainfold trajectories
                    stable (bool): if True, returns stable mainfold IC's, else
                        returns unstable mainfolds

                Returns:
                    None
        """

        sc = self.create_sc(p_sc)
        self.propogate(tau_f= float(T), eval_STM= True)

        STMs = sc.get_STMs()
        states = sc.get_states().T
        pts = np.linspace(0, len(states), N)
    
        self.clear_scs()
        um_list = np.empty(6)
        sm_list = np.empty(6)
        for pt in pts:
            s0 = states[int(pt)-1]
            sc = self.create_sc(s0)

            self.propogate(tau_f= T, eval_STM= True)
            self.clear_scs()

            um_states, sm_states = tools.manifold_state(sc)
            
            um_list = np.vstack((um_list, um_states))
            sm_list = np.vstack((sm_list, sm_states))
            
        self.clear_scs()
        if stable:
            for sm_sc in sm_list:
                self.create_sc(sm_sc)
        else:
            for um_sc in um_list:
                self.create_sc(um_sc)

    def clear_scs(self,):
        """
            clear_scs:
                clears spacecraft object instances from memory
        
        """

        self.scs.clear()

    def set_e(self, e):
        """
            set_e:
                sets model eccentricity model

                Only for use in ER3BP!
        """

        self.model.char_dict[self.dynamics]['e'] = e
        new_e = self.model.char_dict[self.dynamics]['e']

        print('Model eccentricity set to {}'.format(new_e))

        print('done')

    def set_mu(self, mu):
        """
            set_mu:
                sets primary mass ratio, mu, for model

                mu = m2/(m1+m2)
        """

        self.model.char_dict[self.dynamics]['mu'] =  mu
        new_mu = self.model.char_dict[self.dynamics]['mu']

        self.mu, self.l_st, self.t_st = self.model.get_char_vals()[0:3]
        pstates_func = self.model.get_pstates_func()
        
        # State of the earth and moon... this may change for other models?
        self.e_state_syn, self.m_state_syn  = pstates_func(0)

        print('Model mu set to {}'.format(new_mu))
        
    def propogate_func(self, sc, Dtau, n_outputs, tol, eval_STM, p_event, event_func = None, epoch = False):
        """
            propogate_func:
                function for use in the simulation.propogate_parallel, 
                    simulation.propogate methods

                    Args:
                        sc (sc object): sc object for propogation
                        tau_f (float): final non-dim time
                        n_outputs (int): number of integration outputs
                                        (NOT # of integration steps!) 
                        tol (tuple): absolute and relative intergation tolerance, set to
                            [1e-12, 1e-10] for high precision
                        eval_STM: if True, propogates STM matrix
                        p_event: tuple specifing state index and trigger value for poincare map
                        event_func: func to record events

                    Returns:
                        s_new (6xN): sc states at each t_eval
                        t_new (N): evaluated times

                    Opt. Returns:
                        y_hits: states at event triggers
                        t_hits: times at event triggers 
                         
        """

        # Set up integration
        state = sc.get_state()
        tau_0 = sc.get_tau()
        t_eval = np.linspace(tau_0, tau_0 + Dtau, n_outputs)

        if eval_STM:
            STM = sc.STM
            state = np.concatenate((state, STM))

        # Numerical integration 
        sol = scipy.integrate.solve_ivp(
            fun = self.state_func,  # Integrtion func
            y0 = state,             # Initial state
            t_span = [tau_0, tau_0 + Dtau], 
            method = 'RK45',       
            t_eval = t_eval,        # Times to output
            max_step = 1e-2,        
            atol = tol[0],
            rtol = tol[1],
            args = [eval_STM, p_event, epoch],
            events = event_func
        )

        # New states and times for sc
        s_new = sol.y
        tau_new = sol.t

        if event_func != None:
            t_hits = sol.t_events
            y_hits = sol.y_events

            return s_new, tau_new, y_hits, t_hits

        return s_new, tau_new
        
    def propogate_parallel(self, Dtau, n_outputs = 1000, tol = [1e-6, 1e-3], 
            eval_STM = True, p_event = None, event_func = None, verbose = True, epoch = False):
        """
            propogate_parallel:
                propogates scs until tau_f in parallel

                    Args:
                        tau_f (float): final non-dim time
                        n_outputs (float): # of integration outputs
                        tol (tuple): absolute and relative intergation tolerance, set to
                            [1e-12, 1e-10] for high precision
                        eval_STM: if True, propogates STM matrix
                        p_event: tuple specifing state index and trigger value for poincare map
                        event_func: func to record events
                        verbose: if True, prints propogation info

                    Returns:
                        None

        """
        # Create multiprocessing pool
        pool = mp.Pool(os.cpu_count())
        prop_partial = ft.partial(self.propogate_func, Dtau = Dtau, n_outputs = n_outputs, tol = tol, 
            eval_STM = eval_STM, p_event = p_event, event_func = event_func)

        t0 = time.perf_counter()
        # Mutliprocess the propogate_func
        sc_props = pool.map(prop_partial, self.scs)
        pool.terminate()

        # Assign new states and taus to sc
        for i, sc_prop in enumerate(sc_props):

            s = sc_prop[0][0:6, :]
            taus = sc_prop[1]
            
            self.scs[i].set_state(s)
            self.scs[i].set_tau(taus)

            if eval_STM:
                STMs = sc_prop[0][6:, :]
                self.scs[i].set_STM(STMs)

            if event_func != None:
                events = sc_prop[2]
                etaus = sc_prop[3]

                self.scs[i].set_events(events)
                self.scs[i].set_etaus(etaus)
            
        # Print out computational time and action
        tf = time.perf_counter()
        dt = round(tf-t0, 3)

        if verbose:
            print('{} spacecraft propogated in parallel'.format(len(self.scs)))
            print('Computaion time {}'.format(dt))

    def propogate_series(self, Dtau, n_outputs = 1000, tol = [1e-6, 1e-3], 
            eval_STM = True, p_event = None, event_func = None, verbose = True, epoch = False):
        """ 
            propogate_series: 
                propogates sc in series until t
                
                    Args:
                        tau_f (float): final non-dim time
                        n_outputs (float): # of integration outputs
                        tol (tuple): absolute and relative intergation tolerance, set to
                            [1e-12, 1e-10] for high precision
                        eval_STM: if True, propogates STM matrix
                        p_event: tuple specifing state index and trigger value for poincare map
                        event_func: func to record events
                        verbose: if True, prints propogation info

                    Returns:
                        None
        """

        t0 = time.perf_counter()

        # Propogate sc in seris
        for i, sc in enumerate(self.scs):
            
            sc_prop = self.propogate_func(sc, Dtau, n_outputs, tol, eval_STM, p_event, event_func, epoch)

            s = sc_prop[0][0:6, :]
            taus = sc_prop[1]

            self.scs[i].set_state(s)
            self.scs[i].set_tau(taus)

            if eval_STM:
                STMs = sc_prop[0][6:42, :]
                self.scs[i].set_STM(STMs)
        
            if event_func != None:
                events = sc_prop[2]
                etaus = sc_prop[3]

                self.scs[i].set_events(events)
                self.scs[i].set_etaus(etaus)


        # Print out computational time and action
        tf = time.perf_counter()
        dt = round(tf-t0, 3)

        if verbose:
            print('{} spacecraft propogated in series'.format(len(self.scs)))
            print('Computaion time {}'.format(dt))

    def propogate(self, tau_f, n_outputs = None, tol = [1e-6, 1e-3], eval_STM = True, 
        p_event = None, event_func = None, epoch = False, verbose = True):
        """ 
            propogate: 
                General propogation function, determines optimal propogation type and
                    initiates propogation of sim.scs
                
                    Args:
                        tau_f (float): final non-dim time
                        n_outputs (float): # of integration outputs
                        tol (tuple): absolute and relative intergation tolerance, set to
                            [1e-12, 1e-10] for high precision
                        eval_STM: if True, propogates STM matrix
                        p_event: tuple specifing state index and trigger value for poincare map
                        event_func: func to record events
                        verbose: if True, prints propogation info

                    Returns:
                        None

        """

        # Decide to prop in parallel or series
        if len(self.scs)<4:
            propogate_func = self.propogate_series
            if verbose:
                print('Propogating in series..')
        else:
            propogate_func = self.propogate_parallel
            if verbose:
                print('Propogating in parallel..')

        if n_outputs== None:
            n_outputs = int(np.abs(tau_f * 500))

        # Propogate sc 
        propogate_func(tau_f, n_outputs, tol, eval_STM, p_event, event_func, verbose, epoch) 

    def calc_FTLE(self):
        """
            calc_FTLE:
                calculates Finite Time Lyapunov Exponents for 
                sc trajectories in memory

                Uses dyn_sys_tools.py
        """

        print('Beginning FTLE calculation..')
        t0 = time.perf_counter()
        bar = self.progress_bar(len(self.scs))
        bar.start()
        for i, sc in enumerate(self.scs):
            tools.FTLE(sc, self.dynamics)
            bar.update(i+1)

        tf = time.perf_counter()
        dt = round(tf-t0, 3)
        print('FTLEs Calculated..')
        print('Compuation time: {}'.format(dt))

    def calc_STMs(self):

        print('Beginning STM calculation..')
        t0 = time.perf_counter()
        bar = self.progress_bar(len(self.scs))
        bar.start()
        for i, sc in enumerate(self.scs):
            tools.STMs(sc)
            bar.update(i+1)

        tf = time.perf_counter()
        dt = round(tf-t0, 3)
        print('STMs Calculated..')
        print('Compuation time: {}'.format(dt))

    def calc_JC(self):
        """
            calc_JC:
                calculates Jacobi Constant value for 
                sc trajectories in memory

                Uses models.py
        """

        print('Beginning JC calculation..')
        t0 = time.perf_counter()
        # Create multiprocessing pool
        pool = mp.Pool(os.cpu_count())
        JCs_partial = ft.partial(tools.JCs, model = self.model)

        # Mutliprocess the propogate_func
        sc_props = pool.map(JCs_partial, self.scs)
        pool.terminate()

        

        #raise Exception('Testing')

        for i, sc_prop in enumerate(sc_props):
            JCs = sc_prop
            self.scs[i].set_JCs(JCs)

        # bar = self.progress_bar(len(self.scs))
        # bar.start()
        # for i, sc in enumerate(self.scs):
        #     tools.JCs(sc, self.model)
        #     bar.update(i+1)

        tf = time.perf_counter()
        dt = round(tf-t0, 3)
        print('JCs Calculated..')
        print('Compuation time: {}'.format(dt))
 
    def calc_poincare(self, tau_f, p_event):
        """ calc_poincare - calculates Poincaré map of current spacecrafts

                Arguments:
                    tau_f (flt): final ndim time
                    p_event (tuple) [n, t]:
                        n - element in state vector 
                        t - element value to trigger event 

                Example:
                    I want to record x-z plane crossings. Therefore, I need
                    to record when the y component is zero.

                    n : y is the 1 index in the state vector
                    t : the trigger value should be set to 0

                    p_event = [1, 0]

        """

        self.propogate(tau_f, eval_STM = False, p_event = p_event, n_outputs= int(1e4),
            event_func = tools.poincare, tol = [1e-12, 1e-10])

        # if save:
        #     JC_func = self.model.get_JC_func()
        #     Nsc = len(self.scs)
        #     Ttot = tau_f
            
        #     S_inits = np.ndarray((Nsc))
        #     Events = np.ndarray((Nsc), )

        #     for i, sc in enumerate(self.scs):
        #         print(sc.get_events()[0].T)
        #         for event in sc.get_events()[0]

        #         S_inits[i, :] = sc.get_states().T[0]

            #JC = JC_func(s0) ]
         
    def plot_orbit(self, l_dim = False, t_dim = False, inertial = False, Lpoints = False, 
        lims = [1.5, 1.5, 1.5], zvc_JC = None, JC = None, FTLE = False, tau_f = None, ax = None, sun_scale = True, D2 = False,
        D2_axis = None, cont = False, fig = None, e = None, labels = False, cbar = True, zvc_res = 1000):
        """
            plot_orbit:
                plots all sc motion in memory in specified format

                    Opt Args:
                        l_dim (bool): Length dimension
                        t_dim (bool): Time dimension
                        inertial (bool): Inertial coord. frame
                        lims (1x3): sets non-dim frame size [x, y, z]
                        FTLE (bool): if True, displays FTLE values along trajectory
                        tau_f (flt): Final epoch to plot to-- mainly for gif generation
                        scale (bool): if False, reduces scale of Sun in BCR4BP

                    Returns:
                        None
        """
        # if tau_f == None:
        #     tau_f = self.scs[0].get_tau()
        plt.rcParams.update({'font.sans-serif': 'Helvetica'})
        s = 1
        if sun_scale == False:
            s = 80

        pstates_func = self.model.get_pstates_func()
        Lstates_func = self.model.get_Lstates_func()

        # Set dim/non-dim quants
        if l_dim:
            l = self.l_st
            label = '[km]'
        else:
            l = 1 
            label = '[ndim]'
        if t_dim:
            t = self.t_st
        else:
            t = 1

        # Initialize figure object
        if ax == None:
            if D2 == True:
                fig = plt.figure(figsize= [8, (lims[1]/lims[0])*8])
                ax = fig.add_subplot()
                if D2_axis == None:
                    D2_axis = ['x', 'y']
                p = self.axis_dict[D2_axis[0]]
                q = self.axis_dict[D2_axis[1]]

            else:
                fig = plt.figure(figsize= [8, 8])
                ax = fig.add_subplot(projection = '3d')
            #fig.tight_layout()

        # ### Take out
        if D2_axis == None:
            D2_axis = ['x', 'y']
        p = self.axis_dict[D2_axis[0]]
        q = self.axis_dict[D2_axis[1]]

        # Plot each sc
        for i, sc in enumerate(self.scs):

            if tau_f == None:
                tau_f = sc.get_tau()
            mask = sc.get_taus() <= np.abs(tau_f)
            tau_0 = sc.get_taus()[0]
            
            # Inertial coordinate plotting
            if inertial:
                # Create state arrays to fill
                states_syn = sc.get_states()[:, mask]
                states = np.zeros_like(states_syn)
                m_states = np.zeros_like(states_syn)
                e_states = np.zeros_like(states_syn)
                s_states = np.zeros_like(states_syn)

                if Lpoints: # Set to record libration point states
                    L_states = np.zeros((5, 6, len(states_syn[0])))
                
                # Transform each state at its final epoch

                # print(sc.get_taus()[mask])
                for j, tau_f in enumerate(sc.get_taus()[mask]):

                    # Calc tranformation matrix
                    Q = self.transform(tau_0, tau_f)
                    if self.dynamics == 'CR3BP' or self.dynamics == 'ER3BP':
                        e_state_syn, m_state_syn = pstates_func(tau_f)
                    if self.dynamics == 'BCR4BP':
                        e_state_syn, m_state_syn, s_state_syn = pstates_func(tau_f)
                    L_states_syn = Lstates_func(tau_f)

                    # Apply rotation matrix on syn frame state
                    states[:, j] = Q @ states_syn[:, j]
                    e_states[:, j] = (Q @ e_state_syn)*l
                    m_states[:, j] = (Q @ m_state_syn)*l
                    if self.dynamics == 'BCR4BP':
                        s_states[:, j] = (Q @ s_state_syn/s)*l
                    if Lpoints:
                        for k, L in enumerate(L_states):
                            L[:, j]= (Q @ L_states_syn[k])*l

                    # Plots 3d wireframe at last epoch and first sc
                    if i == 0 and j == len(sc.get_taus()[mask]) - 1: 
                        e_state = e_states[:, j]/l
                        m_state = m_states[:, j]/l
                        if D2:
                            earth = self.__earth_2d(e_state, axis = D2_axis, l = l)
                            moon = self.__moon_2d(m_state, axis = D2_axis, l = l)
                            ax.add_artist(earth)
                            ax.add_artist(moon)
                        else:
                            moon = self.__moon_3d(m_state)*l
                            earth = self.__earth_3d(e_state)*l
                            ax.plot_wireframe( moon[0],  moon[1],  moon[2], color = 'dimgray')
                            ax.plot_wireframe(earth[0], earth[1], earth[2], color = 'mediumblue')
                        if self.dynamics == 'BCR4BP':
                            s_state = s_states[:, j]/l
                            if D2:
                                sun = self.__sun_2d(s_state/s, axis = D2_axis, l = l)
                                ax.add_artist(sun)
                            else:
                                sun = self.__sun_3d(s_state/s)*l
                                ax.plot_wireframe(sun[0], sun[1], sun[2], color = 'darkorange')

                        if Lpoints:
                            if D2:
                                for L in L_states:
                                        ax.scatter(L[p, j], L[q, j], color = 'black', s = 3, marker = 'd')
                            else:
                                for L in L_states:
                                    ax.scatter(L[0, j], L[1, j], L[2, j], color = 'black', s = 3, marker = 'd')
                    
                # Line plot the motion of the Primaries
                if D2:
                    ax.plot(e_states[p], e_states[q], color = 'mediumblue', lw = .5)
                    ax.plot(m_states[p], m_states[q], color = 'dimgray', lw = .5)
                else:
                    ax.plot(e_states[0], e_states[1], e_states[2], color = 'mediumblue', lw = .5)
                    ax.plot(m_states[0], m_states[1], m_states[2], color = 'dimgray', lw = .5)
                if self.dynamics == 'BCR4BP':
                    if D2:
                        ax.plot(s_states[p], s_states[q], color = 'darkorange', lw = .5)
                    else:
                        ax.plot(s_states[0], s_states[1], s_states[2], color = 'darkorange', lw = .5)
        
            else:
                # Set states array and plot E and M in syn frame
                states = sc.get_states()[:, mask]
                if self.dynamics == 'CR3BP' or self.dynamics == 'ER3BP':
                    e_state_syn, m_state_syn = pstates_func(tau_f)
                if self.dynamics == 'BCR4BP':
                    e_state_syn, m_state_syn, s_state_syn = pstates_func(tau_f)
                L_states_syn = Lstates_func(tau_f)

                if D2:
                    if self.mu == 0.0121505856:
                        earth = self.__earth_2d(e_state_syn, axis = D2_axis, l = l)
                        moon = self.__moon_2d(m_state_syn, axis = D2_axis, l = l)
                        ax.add_artist(earth)
                        ax.add_artist(moon)

                    else: 
                        m1 = self.__sun_2d(e_state_syn, axis = D2_axis, l = l, s = s*(1-self.mu)*1e2)
                        m2 = self.__sun_2d(m_state_syn, axis = D2_axis, l = l, s = s*(self.mu)*1e2)
                        ax.add_artist(m1)
                        ax.add_artist(m2)
            
                else:
                    earth = self.__earth_3d(e_state_syn)*l
                    moon = self.__moon_3d(m_state_syn)*l
                    ax.plot_wireframe(earth[0], earth[1], earth[2], color = 'mediumblue')
                    ax.plot_wireframe(moon[0],  moon[1],  moon[2], color = 'dimgray')
                if self.dynamics == 'BCR4BP':
                    if D2:
                        print(s)
                        sun = self.__sun_2d(s_state_syn/s, axis = D2_axis, l = l, s = s)
                        ax.add_artist(sun)
                    else:
                        sun = self.__sun_3d(s_state_syn/s)*l
                        ax.plot_wireframe(sun[0], sun[1], sun[2], color = 'darkorange')

                # Plots libration points
                if Lpoints:
                    if D2:
                        for L in L_states_syn:
                            ax.scatter(L[p]*l, L[q]*l, color = 'black', s = 3, marker = 'd')
                    else:
                        for L in L_states_syn:
                            ax.scatter(L[0]*l, L[1]*l, L[2]*l, color = 'black', s = 3, marker = 'd')

            # unpack states array 
            x, y, z = states[0:3]*l

            if sc.monte == True:
                lw = .5
                alpha = 1
                color = 'dimgray'
                m = '.'
            else:
                lw = 2
                alpha = 1
                color = 'firebrick'
                m = '.'

            if FTLE == False and cont == False:
                # Plot light color for monte scs
                if D2:
                    dim1 = states[p]*l 
                    dim2 = states[q]*l
                    ax.plot(dim1, dim2, color = color, alpha = alpha, lw = lw)
                    ### bring back!
                    #ax.scatter(dim1[-1], dim2[-1], color = 'black', s = 0.5)
                else:
                    ax.plot(x, y, z, color = color, alpha = alpha, lw = lw)
                    ax.scatter(x[-1], y[-1], z[-1], color = 'black', s = 0.5)

            elif FTLE == True:
                FTLEs = sc.FTLEs[mask]

                vmin = np.min(self.scs[0].FTLEs[:-5],)
                vmax = np.max(self.scs[0].FTLEs)
                if D2:
                    dim1 = states[p]*l 
                    dim2 = states[q]*l
                    p = ax.scatter(dim1, dim2, c = FTLEs, s = 1, marker = m,
                        cmap = 'inferno', vmin = vmin, vmax = vmax)
                else:
                    p = ax.scatter(x, y, z, c = FTLEs, s = 1, marker = m,
                        cmap = 'inferno', vmin = vmin, vmax = vmax)
                if i == 0 and cbar == True:
                    fig.colorbar(p, pad = .05, shrink = 0.5, orientation = 'vertical',
                    label = r'$ \sigma_{t_0}^{t} $ [ndim] ')

            elif cont == True:
                vmin = 0
                vmax = 0.0549
                dim1 = states[p]*l 
                dim2 = states[q]*l

                print(e)
                p = ax.scatter(dim1, dim2, c = e*np.ones_like(dim1), s = 1, marker = m,
                    cmap = 'cividis_r', vmin = vmin, vmax = vmax)


        # Setting plot attributes
        xlim = lims[0]*l
        ylim = lims[1]*l
        zlim = lims[2]*l

        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)


        if D2:
            ax.set_aspect(1)
            ax.set_xlabel('{} {}'.format(D2_axis[0], label), fontsize = 14)
            ax.set_ylabel('{} {}'.format(D2_axis[1], label), fontsize = 14)

            if zvc_JC != None:
                if JC == None:
                    JC_func = self.model.get_JC_func()
                    s = states.T[0]
                    JC = JC_func(tau_f, s)

                levels = JC + np.array([0, 1e-2])
                xlim = lims[0]
                ylim = lims[1]
                x = np.linspace(-xlim, xlim, zvc_res)
                y = np.linspace(-ylim, ylim, zvc_res)
                X, Y = np.meshgrid(x, y)
                Ust = 2*(0.5*(X**2 + Y**2) + (1-self.mu)/np.sqrt((X + self.mu)**2 + Y**2 ) + self.mu/np.sqrt((X - (1-self.mu))**2 + Y**2))
                ax.contourf(X, Y, Ust, levels = levels, cmap = 'gray')
                ax.annotate(text = 'JC = {}'.format(round(JC, 5)), xy = [-1, 1.2], size = 12)

        if D2 == False:
            ax.set_xlabel('x {}'.format(label))
            ax.set_ylabel('y {}'.format(label))

            ax.set_zlim(-zlim, zlim)
            ax.set_zlabel('z {}'.format(label))
            ax.set_box_aspect([1, ylim/xlim, zlim/xlim])


        if t_dim:
            ax.set_title('{} | T: {} days'.format(self.dynamics, 
                round(tau_f*t/(3600*24), 2)), fontsize = 14)
        else:
            ax.set_title('{} | T: {} [ndim/2π]'.format(self.dynamics,
                round(tau_f/(2*np.pi), 2)), fontsize = 14)


        if labels:
            Llist = ['L1', 'L2', 'L3', 'L4', 'L5']
            for i, L in enumerate(L_states_syn):
                ax.annotate(Llist[i], xy = (L[0]*l, L[1]*l+ 0.05*l), ha = 'center', fontsize = 12)

            Plist = ['Earth', 'Moon', 'Sun']
            for i, prim in enumerate(pstates_func(tau_f)):
                ax.annotate(Plist[i], xy = (prim[0]*l, prim[1]*l+ 0.05*l), ha = 'center', fontsize = 12)


        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        #fig.tight_layout()

        return p 

    def plot_primaries(self, tau_f = None):
        """
            plot_primaries:
                plots 2D orbital motion of primaries -
                mainly used for model analysis
        
        """

        if tau_f == None:
            taus = self.scs[0].get_taus()
            states = self.scs[0].get_states()
        else:
            taus = np.linspace(0, tau_f, 100)

        mu = self.model.char_dict[self.dynamics]['mu']
        shift_mu = np.array([mu, 0, 0, 0, 0, 0])

        m_states = np.ndarray((6, len(taus)))
        e_states = np.ndarray((6, len(taus)))

        pstates_func = self.model.get_pstates_func()
        for i, tau in enumerate(taus):
            Q = self.transform(taus[0], tau)
            det = np.linalg.det(Q)
            print(det)
            e_state, m_state = pstates_func(tau)

            m_states[:, i] = Q @ (m_state) 
            e_states[:, i] = Q @ (e_state)

        # l = self.l_st
        # t = 3.758e5
        
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.plot(m_states[0], m_states[1], color = 'gray')
        ax.plot(e_states[0], e_states[1], color = 'blue')

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.ylabel('y [ndim] (inert)')
        plt.xlabel('X [ndim] (inert)')

        ax.set_aspect('equal', adjustable='box')
        plt.grid()
        plt.tight_layout()

    def plot_poincare(self, axis, projection = None, m = ',', save = False):
        """ 
            plot_poincare:
                Plots poincare map from existing sc propogations

                Arguments:
                    axis (str list): elements of state vector to form axis
                        Options: ['x', 'y', 'z', 'xdot', 'ydot', 'zdot']

                    projection (str): 
                        if '3d', produces 3d plot-- needs 3rd axis

                Example:
                    axis = ['x', 'xdot']
        
        """


        p = self.axis_dict[axis[0]]
        q = self.axis_dict[axis[1]]

        fig = plt.figure(figsize= [5, 5])
        ax = fig.add_subplot()
        
        ax.set_xlabel(axis[0], size = 15)
        ax.set_ylabel(r'$\dot{x}$', size = 15)
        ax.minorticks_on()
        ax.tick_params(which = 'both', direction = 'in')
        #ax.set_ylabel(axis[1])
        if projection == '3d':
            ax = fig.add_subplot(projection = projection)
            w = self.axis_dict[axis[2]]
            ax.set_zlabel(axis[2])

        pstates_func = self.model.get_pstates_func()
        Lstates_func = self.model.get_Lstates_func()
        e_state, m_state = pstates_func(0)
        Lstates = Lstates_func(0)
        plist = []
        qlist = []
        for sc in self.scs:
            estates = sc.get_events()[0].T
            if projection == '3d':
                ax.scatter(estates[p], estates[q], estates[w], s = .2, c = 'black')
            else:
                #sx0 = sc.get_states().T[0, 0]
                #color = np.ones_like(estates[q])*sx0
                #C = ax.scatter(estates[p], estates[q], marker = m, s = 0.1, 
                 #   c = color, vmin = -0, vmax = 1, cmap = 'viridis')
                ax.plot(estates[p], -estates[q], color = 'black', marker = ',', ls = 'none')
                ax.plot(estates[p], estates[q], color = 'black', marker = ',', ls = 'none')

        #fig.colorbar(C)
        ax.scatter(e_state[0], e_state[1], marker = '.', c = 'blue', s = 10)
        ax.scatter(m_state[0], m_state[1], marker = '.', c = 'gray', s = 5)
        for i, L in enumerate(Lstates):
            ax.scatter(L[0], 0, s = 5, marker = '.', color = 'purple')
   
    def plot_FTLE(self, t_dim = False, ax = None):
        """
            plot_FTLE:
                plots the Finite Time Lyapunov Exponent as a function 
                of time for all scs

                Args:
                    t_dim (bool): if True, uses dimensional time
        
        """
        if ax == None:
            # Create figure
            fig = plt.figure(figsize= [5, 5])
            ax = fig.add_subplot()

        if t_dim:
            t = self.t_st
        else:
            t = 1

        for sc in self.scs:

            tau_f = sc.get_tau()
            taus = sc.get_taus()
            FTLE = sc.FTLEs

            if t_dim:
                taus = taus*t/(3600*24)
            else: 
                taus = taus*t/(2*np.pi)

            if sc.monte == True:
                ax.plot(taus, FTLE, color = 'lightcoral', alpha = 1, lw = .5)
            else:
                ax.plot(taus[:-1], FTLE[:-1], color = 'firebrick', alpha = 1, lw = 1)
        
        if t_dim:
            ax.set_xlabel('T [days]')
        else:
            ax.set_xlabel('T [ndim/2π]')

        ax.set_ylabel(r'$ \sigma_{t_0}^{t}$ ')
        ax.set_title('{} '.format(self.dynamics))

    def plot_JC(self, t_dim = False):
        """
            plot_FTLE:
                plots the Finite Time Lyapunov Exponent as a function 
                of time for all scs

                Args:
                    t_dim (bool): if True, uses dimensional time
        
        """

        # Create figure
        fig = plt.figure(figsize= [5, 5])
        ax = fig.add_subplot()

        if t_dim:
            t = self.t_st
        else:
            t = 1

        for sc in self.scs:

            tau_f = sc.get_tau()
            taus = sc.get_taus()
            JCs = sc.get_JCs()

            if t_dim:
                taus = taus*t/(3600*24)
            else: 
                taus = taus*t/(2*np.pi)

            if sc.monte == True:
                ax.plot(taus, JCs, color = 'lightcoral', alpha = 1, lw = .5)
            else:
                ax.plot(taus, JCs, color = 'firebrick', alpha = 1, lw = 1)
        
        if t_dim:
            ax.set_xlabel('T [days]')
        else:
            ax.set_xlabel('T [ndim/2π]')

        ax.set_title('{} '.format(self.dynamics))
        ax.set_ylabel(r'$ JC$ ')

        ax.set_ylim(min(JCs)-1, max(JCs)+1)
        ax.set_xlim(0)
    
        ax.axhline(3.1883, label = 'L1', lw = 0.5, color = 'green')
        ax.axhline(3.1722, label = 'L2', lw = 0.5, color = 'blue')
        ax.axhline(3.0125, label = 'L3', lw = 0.5, color = 'purple')
        ax.axhline(2.9880, label = 'L4/5', lw = 0.5, color = 'turquoise')
        ax.legend()

    def plot_henon(self, JC, density = 1e5):
        xs = np.linspace(-10, 10, int(density))
        x, xdot = tools.henon_curve(x = xs, JC= JC, mu= self.mu)

        plt.plot(x, xdot, ls = 'none', marker = '.', ms  = 1, color = 'black')
        plt.plot(x, -xdot, ls = 'none', marker = '.', ms = 1, color = 'black')

    def movie_orbit(self, propogate, tau_f, frames, fps = 10, gif_name = None, 
        l_dim = False, t_dim = False, inertial = False, L_points = False, 
        lims = [1.5, 1.5, 1.5], FTLE = False, eval_STM = False, sun_scale = False):
        """ movie_orbit - integrates and creates a gif of dynamics in configuration 
            space from the current lead sc time to some tau_f

                Arguments:
                    tau_f (float): the final non-dim time
                    frames (float): # of frames to create, impacts time to render
                    fps (float): frames per second
                    gif_name (str): file name of the gif.. appendend with {}.gif

                    l_dim (bool): if true, plots in dimensional lengths
                    t_dim (bool): if true, plots in dimensional time
                    inertial (bool): if true, plots in inertial coord
                    lims (1x3): non-dim array of of coord limits

                Returns:
                    None

        """

        # Get last time of lead sc and create tau array
        tau_0 = self.scs[0].get_tau() + 1e-3
        taus = np.linspace(tau_0, tau_f, frames)
        
        if propogate:
            # Integrate and plot each time step
            self.propogate(tau_f, eval_STM= eval_STM)
        if FTLE:
            self.calc_FTLE()
        print('Rendering movie..')
        filenames = []
        bar = self.progress_bar(len(taus))
        bar.start()

        for i, tau_f in enumerate(taus):
            print('Movie time')
            print(tau_f)
            self.plot_orbit(l_dim, t_dim, inertial, L_points, lims, FTLE, 
                tau_f = tau_f, sun_scale = sun_scale)
            filename = f'gif_images/image{i}.png'
            filenames.append(filename)
            
            # save frame
            plt.savefig(filename)
            plt.close()
            bar.update(i + 1)


        # build gif
        if gif_name == None:
            today = date.today()
            d1 = today.strftime("%d-%m-%Y")
            gif_name == d1

            print()

        gif_file = 'gif_animations/{}.gif'.format(gif_name)
        with imageio.get_writer(gif_file, mode='I', duration= 1/fps) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                
        # Remove files
        for filename in set(filenames):
            os.remove(filename)

        print('Movie rendered..')

    def __earth_2d(self, r, axis, l):

        R_e = 6371 #km
        p = self.axis_dict[axis[0]]
        q = self.axis_dict[axis[1]]

        r1 = r[p]*l
        r2 = r[q]*l

        radius = (R_e/self.l_st)*l
 
        earth = plt.Circle((r1, r2), radius, color = 'mediumblue')

        return earth

    def __moon_2d(self, r, axis, l):

        R_m = 1737 #km
        p = self.axis_dict[axis[0]]
        q = self.axis_dict[axis[1]]

        r1 = r[p]*l
        r2 = r[q]*l

        radius = (R_m/self.l_st)*l
 
        earth = plt.Circle((r1, r2), radius, color = 'dimgray')

        return earth

    def __sun_2d(self, r, axis, l, s):

        R_s = 695500 #km
        p = self.axis_dict[axis[0]]
        q = self.axis_dict[axis[1]]

        r1 = r[p]*l
        r2 = r[q]*l

        radius = (R_s/self.l_st)*l/s
        sun = plt.Circle((r1, r2), radius, color = 'darkorange')

        return sun

    def __earth_3d(self, r):
        """ 
            __earth_3d:
            produces 3D wireplot state vector of Earth in the synodic frame

                Args:
                    r (1x6 array): position of Earth in ndim synodic frame

                Returns 
                    earth (1x6 array): Earth wireframe state vector
        """

        R_e = 6371 #km

        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
        x = (np.cos(u)*np.sin(v))*R_e/self.l_st + r[0, None]
        y = (np.sin(u)*np.sin(v))*R_e/self.l_st + r[1, None]
        z = (np.cos(v))*R_e/self.l_st + r[2, None]

        earth = np.array([x, y, z, r[3], r[4], r[5]])

        return earth 

    def __moon_3d(self, r):
        """
            __moon_3d:
            produces 3D wireplot state vector of Moon in the synodic frame

                Args:
                    r (1x6 array): position of Moon in ndim synodic frame

                Returns 
                    moon (1x6 array): Moon wireframe state vector
        """
        R_m = 1737 #km

        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
        x = (np.cos(u)*np.sin(v))*R_m/self.l_st + r[0, None]
        y = (np.sin(u)*np.sin(v))*R_m/self.l_st + r[1, None]
        z = (np.cos(v))*R_m/self.l_st + r[2, None]

        moon = np.array([x, y, z, r[3], r[4], r[5]])

        return moon 

    def __sun_3d(self, r):
        """
            __sun_3d:
            produces 3D wireplot state vector of Sun in the synodic frame

                Args:
                    r (1x6 array): position of Sun in ndim synodic frame

                Returns 
                    sun (1x6 array): Sun wireframe state vector
        """
        R_s = 695500 #km

        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
        x = (np.cos(u)*np.sin(v))*R_s/self.l_st + r[0, None]
        y = (np.sin(u)*np.sin(v))*R_s/self.l_st + r[1, None]
        z = (np.cos(v))*R_s/self.l_st + r[2, None]

        sun = np.array([x, y, z, r[3], r[4], r[5]])

        return sun
    
    def progress_bar(self, tot):
        """
            progress_bar:
                progress bar object used for indicating calculation processes
        
        """

        bar = progressbar.ProgressBar(maxval= tot, \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

        return bar

    def single_shooter(self, s0, s_b, T0, T_b, Xd, Xd_b, tol = 1e-7, ax = None):

        """ single_shooter - General single shooting algorithm for state targeting

                Arguments:
                    s0 (1x6 array): initial state 
                    s_b (1x6 array): state boolean array - sets design variables
                    T0 (float): initial propogation time
                    T_b (1x1 array): time boolean array - sets time as design variable
                    
                    Xd (1xN): constraint variables desired values
                    Xd_b (1x6): sets which state elements are the constraints

                Opt Args:
                    tol (flt): tolerance to satisfy for norm of the constrain 
                        function, ||F(X)||
                    ax (mpl axes object): axes object to plot updated trajectory

                Returns:
                    s_i (1x6 array): Solved periodic IC
                    T_i (float): Solved period

                Example:
                    I want to find a L1 Lyapunov orbit with a period of 2.94 ndim
                    using perpendicular crossings
                    
                        # Initial Guesses
                        s0 = [0.8115, 0, 0, 0, 0.255, 0]
                        T0 = 2.94

                        T_b = [False] # Don't change the period
                        Xd  = [0, 0] # Perpendicular crossing requires y_f = 0, vx_f = 0
                        Xd_b = [False, True, False, True, False, False] 

        """

        X_els, FX_els = tools.init_shooter(s_b, T_b, Xd_b)

        # if ax == None:
        #    fig = plt.figure(figsize= [8, 8])
        #    ax = fig.add_subplot(projection = '3d')
        
        # Initial set
        T_i = T0
        s_i = s0
        eps = 1 
        FX_i = 1
        ratio_old = 1
        while eps > tol:
            self.clear_scs()
            sc = self.create_sc(s_i)
            self.propogate(T_i, eval_STM = True, tol = [1e-12, 1e-10])

            X_ip1, FX_ip1 = tools.shooter_update_state(sc, X_els, FX_els, Xd, T_i,
                 self.dynamics)

            for el, X_el in enumerate(X_els):
                if X_el == 6:
                    T_i = X_ip1[el]
                else:
                    s_i[X_el] = X_ip1[el]

            eps = np.linalg.norm(FX_ip1)

            print('Final State Error', eps)
            if eps > 100:
                print('Solution Failed!')
                break

        self.clear_scs()
        self.create_sc(s_i)
        self.propogate(T_i)
        self.calc_FTLE()
        self.plot_orbit(inertial= False, Lpoints= True)

        print('-'*50)
        print('Solved State', s_i)
        print('Solved Period', T_i)
        print('-'*50)
        print()

        return s_i, T_i

    def multi_shooter(self, s0, T0, N = 4, tol = 1e-7):
        """
            multi_shooter:
                multiple shooting algorithm to compute strictly periodic orbits
                with epoch continuity


                ** uses dyn_sys_tools.multi_shooter_update_state()
                ** Currently only takes N = 4 patch points

                ** Current update equation reaches soluion where the first patch point
                begins on the x axis.. [x0, 0, 0, vx, vy, v0]. 
                So be smart while picking your initial guess :) 

                Args:
                    s0 (1x6 array): initial guess of periodic orbit - must be close
                    T0 (float): initial guess of orbit periodi
                    N (int): # of patch points
                    tol (float): tol (flt): tolerance to satisfy for norm of the constrain 
                            function, ||F(X)||

                Returns:
                    S (1x6 array): solved periodic IC 
                    T (float): sovled period of orbit
                    epoch (float): solved initial epoch, must be used in ER3BP, BCR4BP
            
        """

        # Create initial set of states
        sc = self.create_sc(s0)
        self.propogate(T0)

        patches, TOFs = tools.init_patches(sc, N)

        eps = 1
        e_old = 0
        e_new = 1
        self.clear_scs()

        epochs = np.ndarray(N)
        for i in range(len(TOFs)):
            epochs[i] = np.sum(TOFs[0:i])

        while eps > tol:
            
            X_ip1, FX_i = tools.multi_shooter_update_state(self, T0, patches, epochs, TOFs, N, self.dynamics)
            
            patches = np.reshape(X_ip1[0:6*(N)], (N, 6))
            epochs = X_ip1[-N-4:-N]
            TOFs = X_ip1[-N:]

            eps = np.linalg.norm(FX_i)

            ### Optional error computation method
            # eps = np.abs(e_new - e_old)
            # e_old = e_new
            print('Error', eps)
            print()
            print()

            if eps > 0.1:
                return None, None, None, eps

        S = patches[0]
        T = sum(TOFs)
        epoch = epochs[0]

        print('-'*50)
        print('Solved State', S)
        print('Solved Period', T)
        print('Solved Epoch', epoch)
        if self.dynamics == 'ER3BP':
            print('Solved eccentricity', self.model.char_dict[self.dynamics]['e'])
        print('-'*50)
        print()


        return S, T, epoch, eps

    def read_fam_csv(self, path):
        """
            read_fam_csv:
                reads csv file and imports IC and periods
                into numpy arrays

                ** looks in HIDE/families/
                Args:
                    path (str): csv file name

                Returns:
                    states (Nx6 array): family IC
                    periods (N array): family periods corresponding to IC
        """

        df = pd.read_csv('families/{}'.format(path))

        states = df.to_numpy()[7:, 1:7]
        periods = df.to_numpy()[7:,-1]

        return states, periods

    def filter_JC(self,):

        #self.calc_JC() # If you want to scan all JCs -- currently just looks
        # at the last value to see if it is different
        print('Filtering JCs')
        t0 = time.perf_counter()
        Ninit = len(self.scs)
        bar = self.progress_bar(Ninit)
        bar.start()
        JC_func = self.model.get_JC_func()
        for i, sc in enumerate(self.scs):
            taus = sc.get_taus()
            states = sc.get_states()
            s0 = states.T[0]
            sf = states.T[-1]
                
            JC0 = JC_func(taus[0], s0)
            JCf = JC_func(taus[-1], sf)

            resid_norm = np.abs(JCf - JC0)/JC0
            if resid_norm > 5e-4:
                del self.scs[i]
            bar.update(i+1)

        tf = time.perf_counter()
        dt = round(tf-t0, 3)
        Nfinal = len(self.scs)
        sc_rm = Ninit-Nfinal
        self.sc_rm = sc_rm

        print('JCs filtered: {} sc removed'.format(sc_rm))
        print('Compuation time: {}'.format(dt))

    def save_sim(self, states = True, taus = True, 
        FTLEs = False, STMs = False, Events = False, JCs = False):

        mu = self.model.char_dict[self.dynamics]['mu'] 
        e = self.model.char_dict[self.dynamics]['e'] 
        N_sc  = len(self.scs)
        M_eval = len(self.scs[0].get_taus())

        file_name = datetime.now().strftime("%Y-%m-%d_HiDE_Mu{}_e{}_JC{}".format(round(mu, 3), round(e, 3), round(self.JC, 3)))
        file_path = 'save_sim/{}.hdf5'.format(file_name)

        ### Handles multiple saved files for a given day, run
        exists = os.path.exists(file_path)
        i = 0
        while exists == True:
            i += 1
            i_str = str(i)
            temp_name = '_'.join([file_name, i_str])
            temp_path = 'save_sim/{}.hdf5'.format(temp_name)
            exists = os.path.exists(temp_path)
        if i >0:
            file_path = temp_path
            
        # Open file 
        f = h5py.File(file_path, "a")

        # Add general metadata
        f.attrs['mu'] = mu
        f.attrs['e'] = e
        f.attrs['N Ob'] = N_sc
        f.attrs['M Eval'] = M_eval
        f.attrs['JC'] = self.JC
        f.attrs['LMR'] = self.LMR
        f.attrs['density'] = self.density
        f.attrs['tau_0'] = self.tau_0
        f.attrs['sc_rm'] = self.sc_rm

        if states:
            states_grp = f.create_group('States')
            for i, sc in enumerate(self.scs):
                states = sc.get_states().T
                states_grp.create_dataset(name = '{}'.format(i), data = states)
            print('Saved states..')

        if taus:
            taus_grp = f.create_group('Taus')
            for i, sc in enumerate(self.scs):
                taus = sc.get_taus()
                taus_grp.create_dataset(name = '{}'.format(i), data = taus)
            print('Saved taus..')

        if FTLEs:
            FTLE_grp = f.create_group('FTLEs')
            for i, sc in enumerate(self.scs):
                FTLEs = sc.get_FTLEs()
                FTLE_grp.create_dataset(name = '{}'.format(i), data = FTLEs)
            print('Saved FTLEs..')

        if STMs:
            STMs_grp = f.create_group('STMs')
            for i, sc in enumerate(self.scs):
                STMs = sc.get_STMs()
                STMs_grp.create_dataset(name = '{}'.format(i), data = taus)
            print('Saved STMs..')
        
        if Events:
            Events_grp = f.create_group('Events')
            for i, sc in enumerate(self.scs):
                events = sc.get_events()[0]
                Events_grp.create_dataset(name = '{}'.format(i), data = events)
            print('Saved Events..')

        if JCs:
            JCs_grp = f.create_group('JCs')
            for i, sc in enumerate(self.scs):
                JCs = sc.get_JCs()
                JCs_grp.create_dataset(name = '{}'.format(i), data = JCs)
            print('Saved JCs...')

        f.close()

        print('Simulation saved in HDF5 format to...')
        print(file_path)

        return f

        

