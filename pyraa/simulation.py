"""
    simulation module ``simulation.py`` 
    
    Base class of PyRAA

    :Authors: Drew Langford

    :Last Edit: 
        Langford, 07/2022

"""

# Standard packages
from datetime import date
from datetime import datetime
import functools as ft
import multiprocessing as mp
import os
from os import environ
import pathlib
import time
import warnings

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
warnings.filterwarnings("ignore")

# Third-Party packages
import h5py
import imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
import numba as nb
import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.special as sci_sp
import pandas as pd
import progressbar
from pygame import mixer

# PyRAA modules
from pyraa.models import Models
from pyraa.satellite import Satellite
import pyraa.dyn_sys_tools as tools


sys_dict = {
    'Earth-Moon' : {
        'm1' : 5.97e24,
        'm2' : 0.073e24,
        'mu' : 0.0121505856,
        'e' : 0.0549,
        'l_st' : 384748,
        't_st' : 375700,
        'm_st' : 5.97e24 + 0.073e24,
        'p1_c' : 'darkblue',
        'p2_c' : 'dimgray'
    },

    'EarthMoonSun' : {
        'm1' : 5.97e24,
        'm2' : 0.073e24,
        'm3' : 1.989e30,
        'mu' : 0.0121505856,
        'e' : 0.0549,
        'l_st' : 384748,
        't_st' : 375700,
        'a_s' : 149600000/384748,
        'm_s' : 1.989e30/(5.97e24 + 0.073e24)
    },

    'Kepler-34' : {
        'm1' : 1.0479,
        'm2' : 1.0208,
        'mu' : 0.4934,
        'e' : 0.52087,
        't_st' : 1,
        'l_st' : 1,
        'm_st' : 1,
        'p1_c' : None,
        'p2_c' : None
    }
}

class Simulation:
    """
    Simulation -- base class of PyRAA

    The ``Simulation`` class contains the core functionality including 

    * satellite object creation
    * EOM propogation

    Parameters
    ----------
    dynamics: str 
        dynamical model of the simulated enviroment
        
        Options: ['CR3BP', 'ER3BP', 'BCR4BP', 'BER4BP']

    mu: float
        mass ratio of system
    e: float
        eccentricity of system
    system: dict
        pre-set values of a system

    Returns
    -------
        None
    """
    def __init__(self, dynamics, mu = 0.012, e = 0.0, system = None, verbose = True, intro = False):

        if intro:
            file = 'intro_music.mp3'
            mixer.init()
            mixer.music.load(file)
            mixer.music.play()

        self.p1_color = 'darkorange'
        self.p2_color = 'orangered'
        
        if system != None:
            try:
                self.sys_name = system
                self.mu = sys_dict[system]['mu']
            except:
                raise Exception('System not found...')
            else:
                self.e = sys_dict[system]['e']
                self.t_st = sys_dict[system]['t_st']
                self.l_st = sys_dict[system]['l_st']
                self.m_st = sys_dict[system]['m_st']
                
                self.p1_color = sys_dict[system]['p1_c']
                self.p2_color = sys_dict[system]['p2_c']
        else:
            self.sys_name = 'custom'
            self.mu = mu
            self.e = e
            self.t_st = 1
            self.l_st = 1
            self.m_st = 1

        ### Assign dynamical model attributes
        self.dynamics = dynamics
        self.model = Models(self.dynamics)
        self.eom_func = self.model.get_eom_func()
        self.jacobian_func = self.model.get_jacobian_func()

        self.pstates_func = self.model.get_pstates_func()
        self.Lstates_func = self.model.get_Lstates_func()
        
        self.JC_func = self.model.get_JC_func()
        self.transform_func = self.model.get_transform_func()


        # Initialize an empty list of sc objects
        self.sats = []

        # Loose ends 
        self.axis_dict = {
            'x' : 0,
            'y' : 1,
            'z' : 2,
            'xdot' : 3, 
            'ydot' : 4,
            'zdot' : 5
        }

        self.cmap_dict = {
            'cividis' : plt.cm.cividis,
            'viridis' : plt.cm.viridis,
            'jet' : plt.cm.jet,
            'plasma' : plt.cm.plasma
        }

        self.inert_prop_flag = False
        self.prim_prop_flag = False

        if self.e > 0:
            s_max = int(np.log(1e-7)/np.log(self.e) -1)
            self.s = np.arange(1, s_max)
            self.Js = sci_sp.jv(self.s, self.s*e)

        else:
            self.s = None
            self.Js = None

        if intro:
            # Welcome message
            print()
            print('-'*80)
            print('-'*80)
            # print(30*' ', 'WELCOME TO PyRAA')
            print('WELCOME TO PyRAA'.center(80))
            print('Python Restricted Astronomy and Astrodynamics'.center(80))
            time.sleep(1)
            print('-'*80)
            print(f"{5*' '}{'Developer: Drew Langford'}")
            print(f"{5*' '}{'Contact: langfora@purdue.edu'}")
            time.sleep(0.5)
            print('-'*80)
            print()
            logo_file = open("RA_logo.txt")
            lines = logo_file.readlines()
            for line in lines:
                print("{}".format(line[:-1]))
                time.sleep(0.2)
            time.sleep(1)
            logo_file.close()
            print('-'*80)
            print(f"{5*' '}{'Simulation initialized.'}")
            time.sleep(0.2)
            print('-'*80)
            print('-'*80)
            print()

        if verbose:
            self.print_sim_state()
        
        pass

    def print_sim_state(self):
        """
        print_sim_state - prints state of simulation object

        Returns
        -------
        None
        """
        print()
        print('-'*80)
        print('Simulation Overview ')
        print('-'*80)
        print(f"{5*' '}{'System:':<30}{self.sys_name:>30}")
        print(f"{5*' '}{'Dynamics:':<30}{self.dynamics:>30}")
        print(f"{5*' '}{'Mass Ratio (mu):':<30}{self.mu:>30}")
        print(f"{5*' '}{'Eccentricity:':<30}{self.e:>30}")
        print(f"{5*' '}{'Number of sc:':<30}{len(self.sats):>30}")
        print()
        print(f"{5*' '}{'Characteristic Mass [kg]:':<30}{self.m_st:>30}")
        print(f"{5*' '}{'Characteristic Length [km]:':<30}{self.l_st:>30}")
        print(f"{5*' '}{'Characteristic Time [s]:':<30}{self.t_st:>30}")
        print('-'*80)

    def create_sat(self, s0: list, tau_0: float = 0, color: str = None, alpha:float =  1, verbose: bool = False):
        """
        create_sc - creates new satellite object with initial state and epoch

        Parameters
        ----------

        s0: 1x6 array
            initial non-dim state vector of sc
        tau_0: float
            initial non-dim epoch of sc
        color: str or tuple
            color of plotted sc
        alpha: float
            tranparency of plotted sc
        verbose: bool
            if true, prints new sc message
        
        Returns
        -------
        sat: class obj
            sat object instance
        """
        # Initialize a satellite object

        # if color == None:
        #     color = 'firebrick'
        sat = Satellite(s0, tau_0, color = color)
        
        # Add to scs list
        self.sats.append(sat)

        if verbose == True:
            print('New satellite created..')
            print('Total satellite: {}'.format(len(self.sats)))
            print()

        return sat

    def load_sat(self, states = None, taus = None):
        """
        load_sc - loads saved states into memory

        Parameters
        ----------
        states: Nx6 array

        taus: 1xN array
        
        """

        s0 = np.array([0,0,0,0,0,0])
        sat = self.create_sat(s0)

        sat.states = states
        sat.taus = taus


    def ydotconst_func(self, tau, x, jc):
        """
        
        
        
        """

        Ust = 0.5*(x**2) + (1-self.mu)/np.sqrt((x + self.mu)**2)+ self.mu/np.sqrt((x - (1-self.mu))**2)
        ydot = np.sqrt(2*Ust - jc)

        return ydot

    def ydot_griddist(self, x, xdot, JC):
        """
            ydot_griddits - returns ydot values for 
            a given x, xdot, JC on the x-axis

            Used for grid distributions
        
        """

        Vtot = self.Vtot(x, JC)
        Vy = np.sqrt(Vtot**2 - xdot**2)

        return Vy


    def Vtot(self, x, JC):
        
        Ust = 0.5*(x**2) + (1-self.mu)/np.abs(x + self.mu) + self.mu/np.abs(x - (1-self.mu))
        
        Vtot = np.sqrt(2*Ust - JC)
        
        return Vtot

    def Vxy_raddist(self, x, JC, N, rand_v = False):

        """
            xy_raddist - returns a radial distribution of 
                x and y velocities for a given x and JC on the 
                x axis/
        """
        
        if rand_v == True:
            k = np.random.randint(0, N, N)
        else:
            k = np.arange(0, N, 1)

        V = self.Vtot(x, JC)

        # determines radial fanning - larger frac means 
        # larger fan (higher xdot velocities)
        frac = 1/16
        buffer = (1-frac)/2

        xdot = V*np.cos(frac*np.pi*k/N + buffer*np.pi)
        ydot = V*np.sin(frac*np.pi*k/N + buffer*np.pi)

        Xdot = np.concatenate((xdot, xdot))
        Ydot = np.concatenate((ydot, -ydot))
        
        return Xdot, Ydot

    def create_poincare_sc(self, JC, LMR, lim = 1.5, tau_0 = 0, density = 10, 
    plot_ICs = True, monte = False, rand_x = False, rand_v = False, 
        xgrid = None, xdotgrid = None, verbose = False):
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
        if rand_x == True:
            dx = np.abs(- (1+1e-5)*(self.mu) + lim)
            xs_left = - lim + dx*np.random.rand(int(5e4))

            dx = np.abs((1-self.mu)*(1-1e-5) + self.mu*(1-1e-5))
            xs_middle = -self.mu*(1-1e-5) + dx*np.random.rand(int(5e4))
            
            dx = np.abs(lim - (1-self.mu)*(1+1e-5))
            xs_right = (1-self.mu)*(1+1e-5) + dx*np.random.rand(int(5e4))

        else:
            xs_left = np.linspace(-lim, -(1+1e-5)*(self.mu), int(5e4))
            xs_middle = np.linspace(-self.mu*(1-1e-5), (1-self.mu)*(1-1e-5), int(5e4))
            xs_right = np.linspace((1-self.mu)*(1+1e-5), lim, int(5e4))

        fx_left = tools.x_zvc_func(xs_left, self.mu, JC, right = False)
        fx_middle = tools.x_zvc_func(xs_middle, self.mu, JC, right = True)
        fx_right = tools.x_zvc_func(xs_right, self.mu, JC, right = True, left = True)

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
        if np.abs(L) < 1e-3:
            xs_L = np.array([])
        else:
            if np.abs(L - 1) < 1e-3:
                sp_1 = int(np.sum(mask_L)+1/(L))
            else:
                sp_1 = int((np.sum(mask_L)-1)/(L-1))
            xs_L = xs_left[mask_L][0::sp_1]

        if np.abs(M1) < 1e-3:
            xs_M1 = np.array([])
        else:
            left_sum = np.sum(mask_LM)
            right_sum = np.sum(mask_ML)
            sp_1 = int(left_sum/(M1/2))
            sp_2 = int(right_sum/(M1/2))

            xs_M1 = np.concatenate(
                (xs_left[mask_LM][0::sp_1], 
                xs_middle[mask_ML][0::sp_2]
                ))

        if np.abs(M2) < 1e-3:
            xs_M2 = np.array([])
        else:
            left_sum = np.sum(mask_MR)
            right_sum = np.sum(mask_RM)
            sp_1 = int(left_sum/(M2/2))
            sp_2 = int(right_sum/(M2/2))

            xs_M2 = np.concatenate(
                (xs_middle[mask_MR][0::sp_1], 
                xs_right[mask_RM][0::sp_2]
                ))

        if np.abs(R) < 1e-3:
            xs_R = np.array([])
        else:
            if np.abs(R - 1) < 1e-3:
                sp_1 = int(np.sum(mask_R)+1/(R))
            else:
                sp_1 = int((np.sum(mask_R)-1)/(R-1))
            xs_R = xs_right[mask_R][0::sp_1]


        print(xs_L, len(xs_L), 'left')
        print(xs_M1, len(xs_M1), 'M1')
        print(xs_M2, len(xs_M2), 'M2')
        print(xs_R, len(xs_R), 'right')

        
        Xs = np.concatenate((xs_L, xs_M1, xs_M2, xs_R))
        #Xs = np.concatenate((xs_left[mask_left], xs_middle[mask_middle], xs_right[mask_right]))

        N = len(Xs)
        D = 2*density
        xdots = np.ndarray(D*N)
        ydots = np.ndarray(D*N)
        xs = np.ndarray(D*N)
        for n in range(N):
            xdot, ydot = self.Vxy_raddist(Xs[n], JC, density, rand_v)
            xdots[n*D:(n+1)*D] = xdot
            ydots[n*D:(n+1)*D] = ydot
            xs[n*D:(n+1)*D] = Xs[n]

        Odm = np.zeros_like(xs)
        P_ICs = np.concatenate((xs[:,None].T, Odm[:,None].T, 
            Odm[:,None].T, xdots[:,None].T, ydots[:,None].T, Odm[:,None].T)).T
        for state in P_ICs:
            if verbose:
                print('New Poincaré satellite created..')
            self.create_sat(state, tau_0, monte = True)
        print('{} Poincaré sc created..'.format(len(P_ICs)))

        if type(xgrid) != type(None):
            X_grid, Xd_grid = np.meshgrid(xgrid, xdotgrid)

            X_grid_f = X_grid.flatten()
            Xd_grid_f = Xd_grid.flatten()

            ydotgrid = -self.ydot_griddist(X_grid_f, Xd_grid_f, JC)
            print(ydotgrid)
            O_grid = np.zeros_like(X_grid_f)
            grid_ICs = np.concatenate((X_grid_f[:, None].T, O_grid[:,None].T, 
                O_grid[:,None].T, Xd_grid_f[:,None].T, ydotgrid[:,None].T, O_grid[:,None].T)).T
            for state in grid_ICs:
                if verbose:
                    print('New Poincaré satellite created..')
                self.create_sat(state, tau_0, monte = True)
            print('{} Poincaré sc created..'.format(len(grid_ICs)))

        if plot_ICs:
            plt.plot(P_ICs[:, 0], P_ICs[:, 3], ls = 'none', marker = '.', ms = 1, label = str(JC))

            if type(xgrid) != type(None):
                plt.plot(grid_ICs[:, 0], grid_ICs[:, 3], ls = 'none', marker = '.', ms = 1, label = str(JC))
            plt.ylabel(r'$\dot{x}$', size = 14)
            plt.xlabel(r'$x$', size = 14)

            self.plot_henon(JC)
            plt.xlim(np.min(P_ICs[:, 0])-1, np.max(P_ICs[:, 0])+1)
            plt.ylim(np.min(P_ICs[:, 4])-1, np.max(P_ICs[:, 4])+1)
            plt.title(r'$\mu$ = {} | JC = {} '.format(self.mu, JC))

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
        ds_pos = (ds[0:3][:, None] * np.random.randn(3, N))  #/self.l_st
        ds_vel = (ds[3:][:, None] * np.random.randn(3, N))  #*(self.t_st/self.l_st)
        ds = np.concatenate((ds_pos, ds_vel))

        # Add perturbations to model state
        states = (ds + s[:, None]).T

        # Initialize new monte sc
        for state in states:
            print('New monte satellite created..')
            self.create_sat(state, tau_0, monte = True, color = 'dimgray')

    def create_mainifold_sc(self, p_sc, T, N, stable = False):
        """
            create_manifold_sc:
                creates N satellite equally spaced around perodic orbit, which
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

        sc = self.create_sat(p_sc)
        self.propogate(tau_f= float(T), eval_STM= True)

        STMs = sc.get_STMs()
        states = sc.get_states().T
        pts = np.linspace(0, len(states), N)
    
        self.clear_scs()
        um_list = np.empty(6)
        sm_list = np.empty(6)
        for pt in pts:
            s0 = states[int(pt)-1]
            sc = self.create_sat(s0)

            self.propogate(tau_f= T, eval_STM= True)
            self.clear_scs()

            um_states, sm_states = tools.manifold_state(sc)
            
            um_list = np.vstack((um_list, um_states))
            sm_list = np.vstack((sm_list, sm_states))
            
        self.clear_scs()
        if stable:
            for sm_sc in sm_list:
                self.create_sat(sm_sc)
        else:
            for um_sc in um_list:
                self.create_sat(um_sc)

    def clear_sats(self,):
        """
        clear_sats - clears satellite object instances from simulation memory
            
        """

        del(self.sats)

        self.sats = []

    def set_e(self, new_e, verbose = True):
        """
        set_e - sets a new eccentricity value of binary system in er3bp 

        Parameters
        ----------
        new_e: float
            new eccentricity of binary system, [0, 1)
        verbose: bool
            if True, prints confirmation message and sim state
        """

        self.e = new_e

        if verbose:
            print('Model eccentricity set to {}'.format(self.e))
            self.print_sim_state()


    def set_mu(self, new_mu, verbose = True):
        """
        set_mu - sets new mass ratio between p1 and p2

        mu := m1/(m1+m2)

        Parameters
        ----------
        mu: float
            mass ratio of primaries, (0, 0.5]
        verbose: bool
            if True, prints confirmation message and sim state
        """

        self.mu = new_mu

        if verbose:
            print('Model mu set to {}'.format(new_mu))
            self.print_sim_state()
            
    def propogate_func(self, sc, Dtau: float, n_outputs: int, tol: tuple, method: str, 
            eval_STM: bool, p_event: tuple, event_func = None, epoch = False):
        """
        propogate_func - base propogation function

        Function used in `simulation.propogate_parallel` and `simulation.propogate_series`

        Implements `scipy.integrate.solve_ivp`

        Parameters
        ----------
        sat: satellite.Satellite object
            sat object for propogation
        Dtau: float
            time of propogation
        n_outputs: int
            number of integration step outputs - *NOT* number of integration steps! 
        method: str
            method of integration, Options ['RK45', 'DOP853']     
        tol: tuple
            absolute and relative intergation tolerance, set to [1e-12, 1e-10] for high precision
        eval_STM: bool
            if True, propogates STM matrix
        p_event: tuple
            tuple specifing state index and trigger value for poincare map
        event_func: function
            function to record events

        Returns
        -------
        s_new: 6/42xN array 
            sat states at each t_eval
        t_new: 1xN array
            t_evals of the states
        y_hits: array, optional
            states at event triggers
        t_hits: array, optional
            times at event triggers 
        """

        ## Set analytic propogation flags
        self.inert_prop_flag = False
        self.prim_prop_flag = False

        # Set up integration
        state = sc.get_state()
        tau_0 = sc.get_tau()
        t_eval = np.linspace(tau_0, tau_0 + Dtau, n_outputs)

        if eval_STM:
            STM = sc.STM
            state = np.concatenate((state, STM))

        # Numerical integration 
        sol = scipy.integrate.solve_ivp(
            fun = self.eom_func,  # Integrtion func
            y0 = state,             # Initial state
            t_span = [tau_0, tau_0 + Dtau], 
            method = method,  #'DOP853', #    
            t_eval = t_eval,        # Times to output
            max_step = 1e-3,        
            atol = tol[0],
            rtol = tol[1],
            args = [self.mu, self.e, self.s, self.Js, eval_STM, p_event, epoch],
            events = event_func
        )

        # New states and times for sc
        s_new = sol.y[:, 1:]
        tau_new = sol.t[1:]

        if event_func != None:
            t_hits = sol.t_events
            y_hits = sol.y_events

            return s_new, tau_new, y_hits, t_hits

        return s_new, tau_new
        
    def propogate_parallel(self, Dtau: float, n_outputs: int = 1000, tol: tuple = [1e-6, 1e-3], 
            method: str = 'RK45', eval_STM: bool = True, p_event: tuple = None, 
            event_func = None, verbose: bool = True, epoch = False):
        """
        propogate_parallel - parallel propogation method 

        Implements Python multiprocessing to propogate sats in parallel 

        Parameters
        ----------
        Dtau: float
            time of propogation
        n_outputs: int
            number of integration step outputs - *NOT* number of integration steps! 
        tol: tuple
            absolute and relative intergation tolerance, set to [1e-12, 1e-10] for high precision
        method: str
            method of integration, Options ['RK45', 'DOP853']     
        eval_STM: bool
            if True, propogates STM matrix
        p_event: tuple
            tuple specifing state index and trigger value for poincare map
        event_func: function
            function to record events
        verbose: bool
            if True, prints propogation report

        Returns
        -------
        None
        """
        # Create multiprocessing pool
        pool = mp.Pool(os.cpu_count())
        prop_partial = ft.partial(self.propogate_func, Dtau = Dtau, n_outputs = n_outputs, tol = tol, 
            eval_STM = eval_STM, method = method, p_event = p_event, event_func = event_func)

        t0 = time.perf_counter()
        # Mutliprocess the propogate_func
        sc_props = pool.map(prop_partial, self.sats)
        pool.terminate()

        # Assign new states and taus to sc
        for i, sc_prop in enumerate(sc_props):

            s = sc_prop[0][0:6, :]
            taus = sc_prop[1]
            self.sats[i].set_state(s)
            self.sats[i].set_tau(taus)

            if eval_STM:
                STMs = sc_prop[0][6:, :]
                self.sats[i].set_STM(STMs)

            if event_func != None:
                events = sc_prop[2]
                etaus = sc_prop[3]

                self.sats[i].set_events(events)
                self.sats[i].set_etaus(etaus)
            
        # Print out computational time and action
        tf = time.perf_counter()
        dt = round(tf-t0, 3)

        if verbose:
            print('{} satellite propogated in parallel'.format(len(self.sats)))
            print('Computaion time {}'.format(dt))

    def propogate_series(self, Dtau: float, n_outputs: int = 1000, tol: tuple = [1e-6, 1e-3], 
            method: str = 'RK45', eval_STM: bool = True, p_event: tuple = None, 
            event_func = None, verbose: bool = True, epoch = False):
        """ 
        Parameters
        ----------
        Dtau: float
            time of propogation
        n_outputs: int
            number of integration step outputs - *NOT* number of integration steps! 
        tol: tuple
            absolute and relative intergation tolerance, set to [1e-12, 1e-10] for high precision
        method: str
            method of integration, Options ['RK45', 'DOP853']     
        eval_STM: bool
            if True, propogates STM matrix
        p_event: tuple
            tuple specifing state index and trigger value for poincare map
        event_func: function
            function to record events
        verbose: bool
            if True, prints propogation report

        Returns
        -------
        None
        """

        t0 = time.perf_counter()

        # Propogate sc in seris
        for i, sc in enumerate(self.sats):

            sc_prop = self.propogate_func(sc, Dtau, n_outputs, tol, method, eval_STM, p_event, event_func, epoch)

            s = sc_prop[0][0:6, :]
            taus = sc_prop[1]

            self.sats[i].set_state(s)
            self.sats[i].set_tau(taus)

            if eval_STM:
                STMs = sc_prop[0][6:42, :]
                self.sats[i].set_STM(STMs)
        
            if event_func != None:
                events = sc_prop[2]
                etaus = sc_prop[3]

                self.sats[i].set_events(events)
                self.sats[i].set_etaus(etaus)

        # Print out computational time and action
        tf = time.perf_counter()
        dt = round(tf-t0, 3)

        if verbose:
            print('{} satellite propogated in series'.format(len(self.sats)))
            print('Computaion time {}'.format(dt))

    def propogate(self, dtau: float, sat = None, n_outputs: int = None, tol: tuple = [1e-6, 1e-3], 
            method: str = 'RK45', eval_STM: bool = True, p_event: tuple = None, event_func = None, 
            epoch: bool = False, primaries: bool = False, verbose: bool = True):
        """ 
        propogate - general propogation function of PyRAA

        Determines optimal propogation type and initiates propogation of sim.sats
                
        Parameters
        ----------
        dtau: float
            time of propogation
        n_outputs: int
            number of integration step outputs - *NOT* number of integration steps! 
        tol: tuple
            absolute and relative intergation tolerance, set to [1e-12, 1e-10] for high precision
        method: str
            method of integration, Options ['RK45', 'DOP853']     
        eval_STM: bool
            if True, propogates STM matrix
        p_event: tuple
            tuple specifing state index and trigger value for poincare map
        event_func: function
            function to record events
        primaries: bool
            if True, propogates primary positions
        verbose: bool
            if True, prints propogation report

        Returns
        -------
        None
        """

        ## Place to add in specific sat propogation capability
        if sat == None:
            # Decide to prop in parallel or series
            if (len(self.sats) < 2) or (len(self.sats) * dtau < 10):
                propogate_func = self.propogate_series
                if verbose:
                    print('Propogating in series..')
            else:
                propogate_func = self.propogate_parallel
                if verbose:
                    print('Propogating in parallel..')

            if n_outputs== None:
                n_outputs = int(np.abs(dtau * 500))

            # Propogate sc 
            propogate_func(dtau, n_outputs, tol, method, eval_STM, p_event, event_func, verbose, epoch) 

        if primaries:
            self.prop_primaries()

    def prop_periodic(self, S0, T, N):
        """
        prop_periodic - resets S0 after T, N times
        
        A "cheater" function to generate periodic orbits over long time intervals
        when precision, time, and motivation are running low :)

        Parameters
        ----------
        S0: 1x6 array
            Initial periodic state array
        T: float
            Period 
        N: int
            number of periods to repeat


        """

        sat = self.create_sat(S0)
        for iter in range(N):
            self.propogate(T)
            s_f = sat.states
            err = np.linalg.norm(s_f - S0)
            print('Orbit {:} | Error {:3.7f}'.format(N, err), end = '\r')
            sat.states = S0


    @nb.jit(parallel = True)
    def prop_primaries(self, taus = None, mu = None, e = None):
        """
        prop_primaries - propogates primary masses


        Uses dynamical model to sovle for primary positions and Lagrange points
        given an array of times. Yes, this function is silly for the CR3BP! 


        *Useful for solving Lpoints at a given mu, e!*

        Parameters
        ----------
        taus: float
            array of times to solve for primary positions
        mu: float
            mass ratio of system
        e: float
            eccentricity of system
    
        """

        if self.prim_prop_flag:
            del(self.p1_states) 
            del(self.p2_states)
            del(self.LP_states)

        print('Propogating Primaries and Lagrange Points')
        ## Populate the primary and Lpoint arrays
        pstates_func = self.model.get_pstates_func()
        Lstates_func = self.model.get_Lstates_func()

        mu = self.mu 
        e = self.e
        if type(taus) == type(None):
            states = self.sats[0].get_states().T
            taus = self.sats[0].get_taus()
        if type(e) == type(None):
            e = self.e
        if type(mu) == type(None):
            mu = self.mu

        N = len(taus)
        if self.dynamics == 'CR3BP':
            time_array = taus 
        elif self.dynamics == 'ER3BP':
            time_array = np.zeros_like(taus)
            for i in nb.prange(N):
                time_array[i] = Models.calc_E_series(taus[i], e, self.s, self.Js)
            
        p_states = np.zeros((N, 2, 6))
        L_states = np.zeros((N, 5, 6))
        for i in nb.prange(N):
            # p_states[i] = tools.cr3bp_pstates(taus[i], self.mu)
            L_states[i] = Lstates_func(time_array[i], mu, e)
            p_states[i] = pstates_func(time_array[i], mu, e)

        if self.dynamics == 'CR3BP' or self.dynamics == 'ER3BP':
            self.p1_states = np.array(p_states)[:, 0, :]
            self.p2_states = np.array(p_states)[:, 1, :]

        if self.dynamics == 'BCR4BP':
            self.p1_states = np.array(p_states)[:, 0, :]
            self.p2_states = np.array(p_states)[:, 1, :]
            self.p3_states = np.array(p_states)[:, 2, :]

        self.LP_states = np.array([
            L_states[:, 0, :], L_states[:, 1, :], L_states[:, 2, :], 
            L_states[:, 3, :], L_states[:, 4, :]
        ])

        self.prim_prop_flag = True


    def calc_inertial(self):
        """
            calc_inertial - calculates inertial state vectors for all sc saved in memory
                            and saves them as attributes to sc objects
        
        """

        if self.prim_prop_flag == False:
            print('Propogating Primary States..')
            self.prop_primaries()

        if self.inert_prop_flag:
            for sat in self.sats:
                del(sat.states_inert)
            del(self.p1_inert_states)
            del(self.p2_inert_states)
            del(self.LP_inert_states)

        print('Beginning Inertial calculation..')
        t0 = time.perf_counter()

        # Grab the times to transfrom over
        taus = self.sats[0].get_taus()
        tau0 = taus[0]
        N = len(taus)

        # Set up and execute parallel computation of the rotation matrix
        pool = mp.Pool(os.cpu_count())
        transform_partial = ft.partial(self.transform_func, e = self.e)
        Qs = np.array(pool.map(transform_partial, taus))
        pool.terminate()

        print('Performing Inertial Transformations..')
        for i, sc in enumerate(self.sats):
            states = sc.get_states().T
            inert_states = tools.transform_mult(Qs, states, N) #numba enchanced
            sc.set_inert_states(inert_states[1:].T)
        
        self.LP_inert_states = np.zeros((5, N, 6))

        if self.dynamics == 'CR3BP' or self.dynamics == 'ER3BP':
            self.p1_inert_states = tools.transform_mult(Qs, self.p1_states, N)
            self.p2_inert_states = tools.transform_mult(Qs, self.p2_states, N)

        if self.dynamics == 'BCR4BP':
            self.p1_inert_states = tools.transform_mult(Qs, self.p1_states, N)
            self.p2_inert_states = tools.transform_mult(Qs, self.p2_states, N)
            self.p3_inert_states = tools.transform_mult(Qs, self.p3_states, N)

        for j, L_syn in enumerate(self.LP_states):
            self.LP_inert_states[j]= tools.transform_mult(Qs, L_syn, N)

        tf = time.perf_counter()
        dt = round(tf-t0, 3)
        print('Inertial States Calculated..')
        print('Compuation time: {}'.format(dt))

        self.inert_prop_flag = True

    def calc_OE(self,):
        """
        calc_OE - calculates orbital elements of saved sc states

        ** Will calc inertial first if not yet done
        
        """

        if self.inert_prop_flag == False:
            self.calc_inertial()

        calc_OE = tools.calc_RV2OE

        for sc in self.sats:
            states = sc.inert_states
            OE_states = calc_OE(states)
            sc.oe_states = {
                'f' : OE_states[:, 0],
                'a' : OE_states[:, 1],
                'e' : OE_states[:, 2],
                'i' : OE_states[:, 3],
                'O' : OE_states[:, 4],
                'w' : OE_states[:, 5],
            }

    def calc_spline(self, sc, periodic = False):
        """
        calc_spline - calculates cubic spline of current sc states using 
            scipy.interpolate.CubicSpline

            * Sets sc.spline_tau, sc.spline_coeff, and sc.spline_func attr
        

        Parameters
        ----------
        sc: class obj
            sc object from satellite.satellite
        periodic: bool
            if True, sets spline boundary condition to 'periodic'

        Returns
        -------
        None
        """

        states = sc.get_states().T
        taus = sc.get_taus()

        if periodic:
            bc_type = 'Periodic'
            if np.isclose(states[-1], states[0]).all():
                states[-1] = states[0]
            else:
                raise Error('Initial and Final State not sufficiently periodic')
        else:
            bc_type = None

        cs_obj = scipy.interpolate.CubicSpline(taus, states, bc_type = bc_type)

        sc.spline_tau = cs_obj.x
        sc.spline_coeff = cs_obj.c
        sc.spline_func = cs_obj




    def calc_FTLE(self):
        """
            calc_FTLE:
                calculates Finite Time Lyapunov Exponents for 
                sc trajectories in memory

                Uses dyn_sys_tools.py
        """

        print('Beginning FTLE calculation..')
        t0 = time.perf_counter()
        for i, sc in pbar.progressbar(enumerate(self.sats)):
            tools.FTLE(sc, self.dynamics)
            

        tf = time.perf_counter()
        dt = round(tf-t0, 3)
        print('FTLEs Calculated..')
        print('Compuation time: {}'.format(dt))

    def calc_STMs(self):

        print('Beginning STM calculation..')
        t0 = time.perf_counter()
        bar = self.progress_bar(len(self.sats))
        bar.start()
        for i, sc in enumerate(self.sats):
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
        JCs_partial = ft.partial(tools.JCs, model = self.model, mu = self.mu, e = self.e)

        # Mutliprocess the propogate_func
        sc_props = pool.map(JCs_partial, self.sats)
        pool.terminate()

        #raise Exception('Testing')

        for i, sc_prop in enumerate(sc_props):
            JCs = sc_prop
            self.sats[i].set_JCs(JCs)

        # bar = self.progress_bar(len(self.sats))
        # bar.start()
        # for i, sc in enumerate(self.sats):
        #     tools.JCs(sc, self.model)
        #     bar.update(i+1)

        tf = time.perf_counter()
        dt = round(tf-t0, 3)
        print('JCs Calculated..')
        print('Compuation time: {}'.format(dt))
 
    def calc_poincare(self, tau_f, p_event):
        """ calc_poincare - calculates Poincaré map of current satellites

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
        #     Nsc = len(self.sats)
        #     Ttot = tau_f
            
        #     S_inits = np.ndarray((Nsc))
        #     Events = np.ndarray((Nsc), )

        #     for i, sc in enumerate(self.sats):
        #         print(sc.get_events()[0].T)
        #         for event in sc.get_events()[0]

        #         S_inits[i, :] = sc.get_states().T[0]

            #JC = JC_func(s0) ]
         
    def plot_orbit(self, l_dim = False, t_dim = False, inertial = False, Lpoints = False, 
        lims = [1.5, 1.5, 1.5], zvc = False, zvc_JC = None,  FTLE = False, tau_f = None, ax = None, sun_scale = True, D2 = False,
        D2_axis = None, cont = False, fig = None, e = None, labels = False, cbar = False, zvc_res = 1000,
        v_min = None, v_max = None, arrows = False, num_arrows = 3, arr_size = 0.03, trail = 4*np.pi, 
        reverse_cmap = True, c_array = None):
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

        
        if tau_f == None:
            tau_f = self.sats[0].get_tau()
        
        s = 1
        if sun_scale == False:
            s = 50

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

        if D2_axis == None:
            D2_axis = ['x', 'y']
        p = self.axis_dict[D2_axis[0]]
        q = self.axis_dict[D2_axis[1]]

        # Initialize figure object
        if ax == None:
            # Setting plot attributes
            plt.rcParams.update({'font.sans-serif': 'Helvetica'})

            if D2 == True:
                fig = plt.figure(figsize= [8, (lims[1]/lims[0])*8])
                ax = fig.add_subplot()

                ax.set_aspect(1)
                xlim = lims[0]*l
                ylim = lims[1]*l
                ax.set_xlim(-xlim, xlim)
                ax.set_ylim(-ylim, ylim)
                ax.set_xlabel('{} {}'.format(D2_axis[0], label), fontsize = 14)
                ax.set_ylabel('{} {}'.format(D2_axis[1], label), fontsize = 14)

                ax.set_facecolor("lightgray")

            else:
                fig = plt.figure(figsize= [8, 8], facecolor='lightgray')
                ax = fig.add_subplot(projection = '3d')

                xlim = lims[0]*l
                ylim = lims[1]*l
                zlim = lims[2]*l
                ax.set_xlim(-xlim, xlim)
                ax.set_ylim(-ylim, ylim)
                ax.set_zlim(-zlim, zlim)

                ax.set_xlabel('x {}'.format(label))
                ax.set_ylabel('y {}'.format(label))
                ax.set_zlabel('z {}'.format(label))

                ax.set_box_aspect([1, ylim/xlim, zlim/xlim])

                ax.set_facecolor("lightgray")

        cmap = self.cmap_dict['plasma']
        if reverse_cmap:
            cmap = cmap.reversed()

        if c_array == None:
            colors = cmap(np.linspace(0, 1, len(self.sats)))
        else:
            colors = cmap((c_array - np.min(c_array))/(np.max(c_array)- np.min(c_array)))

        if cbar:
            norm = mpl.colors.Normalize(v_min, v_max)
            map = mpl.cm.ScalarMappable(norm= norm, cmap= 'plasma')

            col_bar = fig.colorbar(mappable= map, ax = ax, fraction=0.046, pad=0.04)



        # Plot each sc

        if self.prim_prop_flag == False:
            print('Propogating Primaries..')
            self.prop_primaries()

        for i, sc in enumerate(self.sats):

            taus = sc.get_taus()
            tau_0 = taus[0]
            if tau_f == None:
                tau_f = sc.get_tau()

            tau_mask = taus <= np.abs(tau_f)

            tau_trail = tau_f - trail
            arg_trail = np.argmin(np.abs(taus - tau_trail))

            # Inertial coordinate plotting
            if inertial:
                # set inert states

                if self.inert_prop_flag == False:
                    print('Propogating Inertial States..')
                    self.calc_inertial()

                sc_states = sc.states_inert[tau_mask][arg_trail:]
                p1_states = self.p1_inert_states[tau_mask][arg_trail:]
                p2_states = self.p2_inert_states[tau_mask][arg_trail:]
                Lstates = self.LP_inert_states[:, tau_mask][:, arg_trail:]

            elif inertial == False:
                # set syn states
                sc_states = sc.states[tau_mask][arg_trail:]
                p1_states = self.p1_states[tau_mask][arg_trail:]
                p2_states = self.p2_states[tau_mask][arg_trail:]
                Lstates = self.LP_states[:, tau_mask][:, arg_trail:]

                # Plots 3d wireframe at last epoch and first sc
            if i == 0: 
                if D2:
                    p1 = self.__pimary_2d(p1_states[-1], axis = D2_axis, mass = 1-self.mu, 
                                        color = self.p1_color, beta = 0.8, l = l, r_scale = 0.1)
                    p2 = self.__pimary_2d(p2_states[-1], axis = D2_axis, mass = self.mu, 
                                        color = self.p2_color, beta = 0.8, l = l, r_scale = 0.1)

                    ax.add_artist(p1)
                    ax.add_artist(p2)
                    ax.plot(p1_states[:, p], p1_states[:, q], 
                            lw = 1, color = self.p1_color, zorder = 0)
                    ax.plot(p2_states[:, p], p2_states[:, q], 
                            lw = 1, color = self.p2_color, zorder = 0)

                    if Lpoints == True:
                        
                        ax.scatter(Lstates[:, -1, p], Lstates[:, -1, q], 
                                    color = 'blueviolet', s = 3, marker = 'd') 

                        Llist = ['L1', 'L2', 'L3', 'L4', 'L5']
                        for L_i, L in enumerate(Lstates):
                            
                            ax.plot(L[:, p], L[:, q],  
                                    color = 'blueviolet', lw = 1, alpha = 0.5)
                            if labels:
                                ax.annotate(Llist[L_i], xy = (L[-1, p]*l, L[-1,  q]*l+ 0.05*l), ha = 'center', fontsize = 12)

                        #if labels:


                    
                elif D2 == False:
                    p1 = self.__primary_3d(p1_states[-1], mass = 1-self.mu, beta = 0.8, l = l, r_scale= 0.1)
                    p2 = self.__primary_3d(p2_states[-1], mass = self.mu, beta = 0.8, l = l, r_scale= 0.1)

                    ax.plot_wireframe(p1[0], p1[1], p1[2], color = self.p1_color)
                    ax.plot_wireframe(p2[0], p2[1], p2[2], color = self.p2_color)

                    ax.plot(p1_states[:, 0], p1_states[:, 1], 
                            lw = 1, color = self.p1_color, zorder = 0)
                    ax.plot(p2_states[:, 0], p2_states[:, 1], 
                            lw = 1, color = self.p2_color, zorder = 0)

                    if Lpoints:
                        ax.scatter(Lstates[:, -1, 0], Lstates[:, -1, 1], Lstates[:, -1, 2], 
                                    color = 'blueviolet', s = 3, marker = 'd')

                        Llist = ['L1', 'L2', 'L3', 'L4', 'L5']
                        for L_i, Lstate in enumerate(Lstates):
                            ax.plot(Lstate[:, 0], Lstate[:, 1], Lstate[:, 2], 
                                    color = 'blueviolet', lw = 1, alpha = 0.5)  
                            if labels:
                                ax.text(Lstate[-1, p]*l, Lstate[-1,  q]*l+ 0.05*l, Lstate[-1,  -1]*l+ 0.05*l, Llist[L_i], color='black', ha = 'center', fontsize = 12)
    

                    ###!!! add back in when systems are implemented
                    # if self.dynamics == 'BCR4BP':
                    #     p3_state = self.p3_inert_states[:, -1]
                        
                    #     if D2:
                    #         sun = self.__sun_2d(s_state/s, axis = D2_axis, l = l)
                    #         ax.add_artist(sun)
                    #     else:
                    #         sun = self.__sun_3d(s_state/s)*l
                    #         ax.plot_wireframe(sun[0], sun[1], sun[2], color = 'darkorange')



            if D2:
                print(sc.color)
                if sc.color == None:
                    color = colors[i]
                else:
                    color = sc.color

                ax.plot(sc_states[:, p]*l, sc_states[:, q]*l, color = color, alpha = sc.alpha, lw = sc.lw, ls = sc.ls, zorder = 0)
                ax.scatter(sc_states[-1, p]*l, sc_states[-1, q]*l, color = 'dodgerblue', edgecolors = 'dodgerblue', marker = sc.marker, zorder = 1)
            elif D2 == False:
                ax.plot(sc_states[:, 0]*l, sc_states[:, 1]*l, sc_states[:, 2]*l, color = colors[i], alpha = sc.alpha, lw = sc.lw, ls = sc.ls, zorder = 0)
                ax.scatter(sc_states[-1, 0]*l, sc_states[-1, 1]*l, sc_states[-1, 2]*l, color = 'dodgerblue', edgecolors = 'dodgerblue', marker = sc.marker, zorder = 0)



        #     if FTLE == False and cont == False:
        #         # Plot light color for monte scs
        #         if D2:
        #             dim1 = states[p]*l 
        #             dim2 = states[q]*l
        #             if type(sc.color) == type(''):
        #                 ax.plot(dim1, dim2, color = sc.color, alpha = sc.alpha, lw = lw)
        #             elif type(sc.color) == type(np.double(1.2)):
        #                 print(sc.color)
        #                 scat = ax.scatter(dim1, dim2, c = sc.color*np.ones_like(dim1), 
        #                     alpha = alpha, s = 0.5, cmap = 'jet', vmin = scat_min, vmax = scat_max, 
        #                     rasterized = True)
        #             ### bring back!
        #             #ax.scatter(dim1[-1], dim2[-1], color = 'black', s = 0.5)
        #             if arrows:
        #                 num_states = len(dim1)
        #                 num_arr_states = int(num_states/num_arrows)
            

        #                 for i in range(num_arrows):
        #                     dstate = 1
        #                     x0 = dim1[1 + i*num_arr_states] 
        #                     x1 = dim1[1 +i*num_arr_states + dstate] 
        #                     y0 = dim2[1 +i*num_arr_states] 
        #                     y1 = dim2[1 +i*num_arr_states + dstate] 
        #                     ax.arrow(x0, y0, x1-x0, y1-y0, width = 0, head_width = arr_size,
        #                     color = 'black', zorder = 3, head_starts_at_zero = False, overhang = 0.1)
        #         else:
        #             ax.plot(x, y, z, color = sc.color, alpha = alpha, lw = lw)
        #             ax.scatter(x[-1], y[-1], z[-1], color = 'black', s = 0.5)

        #     elif FTLE == True:
        #         FTLEs = sc.FTLEs[mask]

        #         vmin = np.min(self.sats[0].FTLEs[:-5],)
        #         vmax = np.max(self.sats[0].FTLEs)
        #         if D2:
                    
        #             dim1 = states[p]*l 
        #             dim2 = states[q]*l
        #             scat = ax.scatter(dim1, dim2, c = FTLEs, s = 1, marker = m,
        #                 cmap = 'inferno', vmin = vmin, vmax = vmax)
        #         else:
        #             scat = ax.scatter(x, y, z, c = FTLEs, s = 1, marker = m,
        #                 cmap = 'inferno', vmin = vmin, vmax = vmax)
        #         if i == 0 and cbar == True:
        #             fig.colorbar(scat, pad = .05, shrink = 0.5, orientation = 'vertical',
        #             label = r'$ \sigma_{t_0}^{t} $ [ndim] ')

        #     elif cont == True:
        #         vmin = 0
        #         vmax = 0.0549
        #         dim1 = states[p]*l 
        #         dim2 = states[q]*l

        #         print(e)
        #         scat = ax.scatter(dim1, dim2, c = e*np.ones_like(dim1), s = 1, marker = m,
        #             cmap = 'cividis_r', vmin = vmin, vmax = vmax)





        # if D2:
        #     if zvc:
        #         if zvc_JC == None:
        #             JC_func = self.model.get_JC_func()
        #             s = states.T[0]
        #             JC = JC_func(tau_f, s)

        #         levels = JC + np.array([0, 1e-2])
        #         xlim = lims[0]
        #         ylim = lims[1]
        #         x = np.linspace(-xlim, xlim, zvc_res)
        #         y = np.linspace(-ylim, ylim, zvc_res)
        #         X, Y = np.meshgrid(x, y)
        #         Ust = 2*(0.5*(X**2 + Y**2) + (1-self.mu)/np.sqrt((X + self.mu)**2 + Y**2 ) + self.mu/np.sqrt((X - (1-self.mu))**2 + Y**2))

        #         ax.contourf(X, Y, Ust, levels = levels, cmap = 'gray')
        #         ax.annotate(text = 'JC = {}'.format(round(JC, 5)), xy = [-1, 1.2], size = 12)



        if t_dim:
            ax.set_title('{} | T: {} days'.format(self.dynamics, 
                round(tau_f*t/(3600*24), 2)), fontsize = 14)
        else:
            ax.set_title('{} | T: {} [ndim/2π]'.format(self.dynamics,
                round(tau_f/(2*np.pi), 2)), fontsize = 14)


        # if labels:
        #     Llist = ['L1', 'L2', 'L3', 'L4', 'L5']
        #     for i, L in enumerate(L_states_syn):
        #         ax.annotate(Llist[i], xy = (L[0]*l, L[1]*l+ 0.05*l), ha = 'center', fontsize = 12)

        #     Plist = ['Earth', 'Moon', 'Sun']
        #     for i, prim in enumerate(pstates_func(tau_f)):
        #         ax.annotate(Plist[i], xy = (prim[0]*l, prim[1]*l+ 0.05*l), ha = 'center', fontsize = 12)


        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        #fig.tight_layout()

        if cbar:
            return fig, ax, col_bar

        return fig, ax

    def plot_primaries(self, tau_f = None):
        """
            plot_primaries:
                plots 2D orbital motion of primaries -
                mainly used for model analysis
        
        """

        if tau_f == None:
            taus = self.sats[0].get_taus()
            states = self.sats[0].get_states()
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
        for sc in self.sats:
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

        # for i, L in enumerate(Lstates):
        #     ax.scatter(L[0], 0, s = 5, marker = '.', color = 'purple')
   
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

        for sc in self.sats:

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

        for sc in self.sats:

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
        lims = [1.5, 1.5, 1.5], FTLE = False, eval_STM = False, sun_scale = False,
        D2 = False, zvc = False, scat_min = None, scat_max = None, cbar = False, labels = False, trail = 2*np.pi):
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
        tau_0 = self.sats[0].get_taus()[0] + 1e-3
        taus = np.linspace(tau_0, tau_f, frames)
        
        if propogate:
            # Integrate and plot each time step
            self.propogate(tau_f, eval_STM= eval_STM, prop_prims= True)
        if FTLE:
            self.calc_FTLE()


        print('Rendering movie..')
        filenames = []
        for i, tau_f in pbar.progressbar(enumerate(taus)):
            self.plot_orbit(tau_f = tau_f, l_dim= l_dim, t_dim = t_dim, inertial= inertial,
            Lpoints= L_points, lims = lims, zvc = zvc, FTLE = FTLE, D2 = D2, v_max = scat_max,
            v_min = scat_min, cbar= cbar, trail = trail, labels= labels)
                
            filename = f'gif_images/image{i}.png'
            filenames.append(filename)
            
            # save frame
            plt.savefig(filename)
            plt.close()


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


    def __pimary_2d(self, state, axis, mass, color, 
                    beta = 0.8, l = 1, r_scale = 1):
        """
            __pimary_2d - generates circle object to be plotted

            Args:
                state (array) : state of the primary
                axis ([str, str]) : axis to plot
                mass (float) : the ndim mass, [0, 1]
                color (str) : color of circle

            Opt Args:
                beta (float) : mass to radius relation, 0.8 for stars
                l (float) : char length factor
                r_scale (float) : radius scaling factor


        """

        p = self.axis_dict[axis[0]]
        q = self.axis_dict[axis[1]]

        radius = np.power(mass, beta)*l*r_scale

        x = state[p]*l
        y = state[q]*l

        pimary_circ = plt.Circle((x, y), radius, color = color)

        return pimary_circ


    def __primary_3d(self, state, mass, beta = 0.8, l = 1, r_scale = 1):
        """
            __pimary_3d:
            produces 3D wireplot state vector of primary in the synodic frame

            Args:
                state (array) : state of the primary
                axis ([str, str]) : axis to plot
                mass (float) : the ndim mass, [0, 1]
                color (str) : color of circle

            Opt Args:
                beta (float) : mass to radius relation, 0.8 for stars
                l (float) : char length factor
                r_scale (float) : radius scaling factor
        """

        radius = np.power(mass, beta)*r_scale

        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]

        x = (np.cos(u)*np.sin(v))*radius + state[0, None]
        y = (np.sin(u)*np.sin(v))*radius + state[1, None]
        z = (np.cos(v))*radius + state[2, None]

        sun = np.array([x, y, z, state[3], state[4], state[5]])*l

        return sun

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

    def __sun_3d(self, r, scale = 1):
        """
            __sun_3d:
            produces 3D wireplot state vector of Sun in the synodic frame

                Args:
                    r (1x6 array): position of Sun in ndim synodic frame

                Returns 
                    sun (1x6 array): Sun wireframe state vector
        """
        #R_s = 695500 #km


        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
        # x = (np.cos(u)*np.sin(v))*R_s/self.l_st + r[0, None]
        # y = (np.sin(u)*np.sin(v))*R_s/self.l_st + r[1, None]
        x = (np.cos(u)*np.sin(v))*scale + r[0, None]
        y = (np.sin(u)*np.sin(v))*scale + r[1, None]
        z = (np.cos(v))*scale + r[2, None]

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

    def single_shooter(self, s0, s_b, T0, T_b, Xd, Xd_b, tol = 1e-7, ax = None, verbose = True, plot = False):

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

        ### Take out later! 
        state_solutions = []
        per_solutions = []
        error_list = []
        
        # Initial set
        T_i = T0
        s_i = s0
        eps = 1 
        FX_i = 1
        ratio_old = 1
        while eps > tol:
            self.clear_scs()
            sc = self.create_sat(s_i)
            self.propogate(T_i, eval_STM = True, tol = [1e-14, 1e-13])

            X_ip1, FX_ip1, DFX_ip1 = tools.shooter_update_state(sc, X_els, FX_els, Xd, T_i,
                 self.dynamics, self.mu, self.e, self.s, self.Js)

            for el, X_el in enumerate(X_els):
                if X_el == 6:
                    T_i = X_ip1[el]
                else:
                    s_i[X_el] = X_ip1[el]

            eps = np.linalg.norm(FX_ip1)

            if verbose:
                print('Final State Error', eps)
                print('State', s_i)
                print('T', T_i)
            # print('State:', s_i)
            # print('Period', 2*T_i)

            # state_solutions.append(s_i)
            # per_solutions.append(2*T_i)
            # error_list.append(eps)
            if eps > 100:
                print('Solution Failed!')
                break

        self.clear_scs()
        sc = self.create_sat(s_i)
        self.propogate(T_i)
 
        if plot:
            self.plot_orbit(inertial= False, Lpoints= True)

        print('-'*50)
        print('Solved State', s_i)
        print('Solved Period', T_i)
        print('-'*50)
        print()

        ## take out later

        M = sc.get_STM()
        detM = np.linalg.det(M)
        print('Monodromy Matrix Determinant', detM)

        return s_i, T_i, DFX_ip1

        # return state_solutions, per_solutions, error_list

    def multi_shooter(self, s0, T0, N = 4, tol = 1e-7):
        """
            multi_shooter:
                multiple shooting algorithm to compute strictly periodic orbits
                with epoch continuity


                ** uses dyn_sys_tools.multi_shooter_update_state()
                ** Currently only takes N = 4 patch points

                ** Current update equation reaches solution where the first patch point
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
        sc = self.create_sat(s0)
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

            if eps > 10:
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
        Ninit = len(self.sats)
        bar = self.progress_bar(Ninit)
        bar.start()
        JC_func = self.model.get_JC_func()
        for i, sc in enumerate(self.sats):
            taus = sc.get_taus()
            states = sc.get_states()
            s0 = states.T[0]
            sf = states.T[-1]
                
            JC0 = JC_func(taus[0], s0)
            JCf = JC_func(taus[-1], sf)

            resid_norm = np.abs(JCf - JC0)/JC0
            if resid_norm > 5e-4:
                del self.sats[i]
            bar.update(i+1)

        tf = time.perf_counter()
        dt = round(tf-t0, 3)
        Nfinal = len(self.sats)
        sc_rm = Ninit-Nfinal
        self.sc_rm = sc_rm

        print('JCs filtered: {} sc removed'.format(sc_rm))
        print('Compuation time: {}'.format(dt))

    def save_sim(self, states = True, taus = True, 
        FTLEs = False, STMs = False, Events = False, JCs = False):

        mu = self.model.char_dict[self.dynamics]['mu'] 
        e = self.model.char_dict[self.dynamics]['e'] 
        N_sc  = len(self.sats)
        M_eval = len(self.sats[0].get_taus())

        file_name = datetime.now().strftime("%Y-%m-%d_HiDE_Mu{}_e{}_JC{:4.3f}".format(round(mu, 3), round(e, 3), round(self.JC, 3)))
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
            for i, sc in enumerate(self.sats):
                states = sc.get_states().T
                states_grp.create_dataset(name = '{}'.format(i), data = states)
            print('Saved states..')

        if taus:
            taus_grp = f.create_group('Taus')
            for i, sc in enumerate(self.sats):
                taus = sc.get_taus()
                taus_grp.create_dataset(name = '{}'.format(i), data = taus)
            print('Saved taus..')

        if FTLEs:
            FTLE_grp = f.create_group('FTLEs')
            for i, sc in enumerate(self.sats):
                FTLEs = sc.get_FTLEs()
                FTLE_grp.create_dataset(name = '{}'.format(i), data = FTLEs)
            print('Saved FTLEs..')

        if STMs:
            STMs_grp = f.create_group('STMs')
            for i, sc in enumerate(self.sats):
                STMs = sc.get_STMs()
                STMs_grp.create_dataset(name = '{}'.format(i), data = taus)
            print('Saved STMs..')
        
        if Events:
            Events_grp = f.create_group('Events')
            Etaus_grp = f.create_group('Event Taus')
            for i, sc in enumerate(self.sats):
                events = sc.get_events()[0]
                etaus = sc.get_etaus()[0]

                print(events)
                print(etaus)
                Events_grp.create_dataset(name = '{}'.format(i), data = events)
                Etaus_grp.create_dataset(name = '{}'.format(i), data = etaus)
            print('Saved Events..')

        if JCs:
            JCs_grp = f.create_group('JCs')
            for i, sc in enumerate(self.sats):
                JCs = sc.get_JCs()
                JCs_grp.create_dataset(name = '{}'.format(i), data = JCs)
            print('Saved JCs...')

        f.close()

        print('Simulation saved in HDF5 format to...')
        print(file_path)

        return f



