"""
    targeting module ``targeting.py`` 
    
    This module contains functionality for differential corrections methods to target a 
    design solution given constraints 

    .. attention::
        Classes inherit ``simulation.Simulation``

    :Authors: 
        Drew Langford

    :Last Edit: 
        Langford, 07/2022

"""
# Standard imports
import os
import time

# Third-party
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as scio
import scipy.integrate

# PyRAA imports
from pyraa.simulation import Simulation
from pyraa.models import Models
import pyraa.dyn_sys_tools as tool


class Targeter(Simulation):
    """
    Targeter Class

    Implements differential corrections algorithms for computing a trajectory that 
    satisfies a set of constraints

    Parameters
    ----------
    Xd_vars: list of strs
        Vector of design variables, ['x', 'y', 'z', 'vx', 'vy', 'vz', 'T']

        *Example* -- Xd_vars = ['x', 'vy', 'T']

    Xc_dict: dict
        Dictionary of constraint expressions, key -> variable, value -> constriant
        
        Key options: ['x', 'y', 'z', 'vx', 'vy', 'vz']

        *Example* -- {'y' : 0, 'vx' : 0}
    dynamics: str
        name of dynamical model, options: "CR3BP", "ER3BP"
    mu: float
        mass ratio of primaries
    e: float
        eccentricity
    eps:
        precision of targeting algorithm

    
    Returns
    -------
    None

    """
    def __init__(self, Xd_vars: list, Xc_dict: dict, dynamics: str, mu: float, e: float, eps: float):

        # Init the parent Simulation class
        Simulation.__init__(self, dynamics=dynamics, mu=mu, e=e, verbose= False)

        self.var_dict = {
            'x' :  0, 
            'y' :  1, 
            'z' :  2,
            'vx' : 3,
            'vy' : 4,
            'vz' : 5,
            'T' : 6
            }

        
        self.Nd = len(Xd_vars) # set length of design variable array
        self.Xd_vars = Xd_vars

        self.Nc = len(Xc_dict) # set length of constriant variable array
        self.Xc_dict = Xc_dict

        if self.Nd < self.Nc:
            raise ValueError('Increase design variables')

        # Set up design variable element array
        self.Xd_els = []
        for var in Xd_vars:
            self.Xd_els.append(self.var_dict[var])

        # init error list
        self.errors = []
        self.error = 1000
        
        # set solution precision
        self.eps = eps


        self.psu_arc = False
        

    def solve(self, S_g: list, T_g: float, verbose: bool = True):
        """
        solve - initiates the differential corrections algorithm

        Parameters
        ----------
        S_g: 1x6 array
            Guess for initial state vector
        T_g: float
            Guess for propogation time
        verbose: bool
            if True, prints output of solver
        
        Returns
        -------
        self.S: 1x6 array
            Solved intitial state vector
        self.T: float
            Solved propogation time


        *Assigns atributes*

        * self.S -- solved state
        * self.T -- solved time of propogation
        
        """

        self.error = 50
        self.errors = []

        Xd = self._calc_Xd(S_g, T_g)
        self.Xds = np.array([Xd])
        while self.error > self.eps:
            sc = self.create_sat(S_g)
            self.propogate(T_g, verbose= False, method= 'DOP853', tol = [1e-12, 1e-10])
            S_g, T_g = self._update_state(sc)
            
            self.clear_sats()

            if verbose:
                print('Error: {}'.format(self.error), end='\r')
                time.sleep(0.1)
                
        # set sovlved state as solution attribute
        self.S = S_g
        self.T = T_g
        
        if verbose:
            print()
            print(80*'-')
            print('Solution Found in {} iterations'.format(len(self.errors)))
            print('S : {}'.format(self.S))
            print('T : {}'.format(self.T))
            print(80*'-')
            print()


        return self.S, self.T


    def calc_midstate(self, q_a: list, q_b: list):
        """
        calc_midstate - calculates average state given two design variable arrays


        Takes the saved solution, self.S, self.T and finds a new state that is between
        a new design variable, q_a and old deisgn variable q_b. This could be used for 
        locating the state of a bifurcation. 

        Parameters
        ----------
        q_a: design variable array #1
        q_b: design variable array #2

        Returns
        -------
        S_m: 1x6 array
            mid-state array
        T_m: float
            mid-propogation time
        |dq|: float
            the norm difference between S_m, T_m and the design variable arrays

        """

        S_a, T_b = self._fill_new_state(self.S, self.T, Xd = q_a)
        S_b, T_b = self._fill_new_state(self.S, self.T, Xd = q_b)

        dq = (q_b - q_a)/2
        q_m = q_a + dq
        S_m, T_m = self._fill_new_state(self.S, self.T, Xd = q_m)

        return S_m, T_m, np.linalg.norm(dq)

    def calc_eigenspec(self, S: list, T: float, half: bool = False, verbose: bool = True):
        """
        calc_eigenspec - returns eigenvalues and vectors of state transition matrix
        matrix

        Function propogates *S* for a time, *T*, and calculates eigenspectrum 
        from last state transition matrix. This function implements a comparative 
        algorithm for sorting the eigenvalues and eigen vectors. The sorting is based on 
        the eigenvalue type and follows the general order:
        [unitary, complex, real] 

    
        Parameters
        ----------
        S: 1x6 array
            initial state vector
        T: float
            time of propogation
        half: bool
            if True, propogates for 2*T*, use-case is perpendicular crossing
        verbose: bool
            if True, prints eigenspectrum results
        
        Returns
        -------
        None


        *Assigns atributes*
        
        * self.M -- monodromy matrix, 6x6 array
        * self.val_array -- eigenvalue array, sorted unitary, complex, real
        * self.vec_arary -- eigenvector array, sorted unitary, complex, real
        * self.nu_array -- nu value array, sorted unitary, complex, real
        * self.PCexp_array -- Poincare exponent array, sorted unitary, complex, real
        * self.num_U -- number of unitary eigenvalues
        * self.num_C -- number of complex eigenvalues
        * self.num_R -- number of real eigenvalues

        """

    
        sat = self.create_sat(S)
        self.propogate(T, verbose= False, method = 'DOP853')

        s0 = sat.get_states().T[0]
        sf = sat.get_states().T[-1]

        per_bool = np.isclose(s0, sf)
        if per_bool.any() == False:
            print(Warning('Solution is not periodic !'))

        # Floquet Multipliers
        self.M = sat.get_STM()
        eig_vals, eig_vecs = np.linalg.eig(self.M)
        eig_vecs[np.abs(eig_vecs[:, :])  < 1e-10] = 0 
        self.clear_sats()

        labels = ['U', 'R', 'C']
        U_els = []
        R_els = []
        C_els = []
        U_vals = []
        R_vals = []
        C_vals = []
        nus = []

        # Classify elements
        for el, val in enumerate(eig_vals):
            print(val)
            unit_cond = np.isclose(val.real, 1, atol = 1e-5)
            complex_cond = np.abs(val.imag) > 1e-3
            real_cond = np.isclose(val.imag, 0, atol= 1e-9)
            print(unit_cond, complex_cond, real_cond)
            if unit_cond:
                U_els.append(el)
            elif real_cond:
                R_els.append(el)
            elif complex_cond:
                C_els.append(el)
            else:
                raise Exception('No classification')
                
            nu = 0.5*(val + 1/val).real
            nus.append(nu)


        num_U = int(0.5*len(U_els))
        num_C = int(0.5*len(C_els))
        num_R = int(0.5*len(R_els))
        
        nus = np.array(nus)
        U_argpairs = []
        R_argpairs = []
        C_argpairs = []

        R_nus = []
        C_nus = []
        for el, nu in enumerate(nus):
            arg_pair = np.sort(np.argsort(np.abs(nu - nus))[0:2])
            if el in U_els:
                if el not in np.array(U_argpairs).ravel():
                    U_argpairs.append(arg_pair)
            elif el in R_els:
                if el not in np.array(R_argpairs).ravel():
                    R_argpairs.append(arg_pair)
                    R_nus.append(nu)
            elif el in C_els:
                if el not in np.array(C_argpairs).ravel():
                    C_argpairs.append(arg_pair)
                    C_nus.append(nu)

  
        if len(R_argpairs) != 0:
            R_argpairs_temp = R_argpairs.copy()
            R_arg_order = np.argsort(R_nus)
            for i in range(len(R_argpairs)):
                arg_el = R_arg_order[i]
                R_argpairs[i] = R_argpairs_temp[arg_el]

        if len(C_argpairs) != 0:
            C_argpairs_temp = C_argpairs.copy()
            C_arg_order = np.argsort(C_nus)
            for i in range(len(C_argpairs)):
                arg_el = C_arg_order[i]
                C_argpairs[i] = C_argpairs_temp[arg_el]

        print(U_argpairs, C_argpairs, R_argpairs)

        U_dict_array = [eig_vals[el].real for el in U_argpairs]
        R_dict_array = [eig_vals[el].real for el in R_argpairs]
        C_dict_array = [eig_vals[el] for el in C_argpairs]

        Uvec_dict_array = [eig_vecs[:, el].real.T for el in U_argpairs]
        Rvec_dict_array = [eig_vecs[:, el].real.T for el in R_argpairs]
        Cvec_dict_array = [eig_vecs[:, el].T for el in C_argpairs]

        print(U_dict_array, C_dict_array, R_dict_array)

        self.val_dict = {
            'U' : U_dict_array,
            'C' : C_dict_array,
            'R' : R_dict_array,
        }
        self.vec_dict = {
            'U' : Uvec_dict_array,
            'C' : Cvec_dict_array,
            'R' : Rvec_dict_array,
        }

        self.val_array = np.zeros((3, 2), dtype = np.complex)
        self.vec_array = np.zeros((3, 2, 6), dtype= np.complex)

        U_iter = 0
        while U_iter < num_U:
            self.val_array[U_iter] = self.val_dict['U'][U_iter]
            self.vec_array[U_iter] = self.vec_dict['U'][U_iter]
            U_iter += 1

        C_iter = 0
        while C_iter < num_C:
            self.val_array[U_iter + C_iter] = self.val_dict['C'][C_iter]
            self.vec_array[U_iter + C_iter] = self.vec_dict['C'][C_iter]
            C_iter += 1

        R_iter = 0
        while R_iter < num_R:
            self.val_array[U_iter + C_iter + R_iter] = self.val_dict['R'][R_iter]
            self.vec_array[U_iter + C_iter + R_iter] = self.vec_dict['R'][R_iter]
            R_iter += 1

        self.PCexp_array = 1/T*np.log(self.val_array)
        self.PCexp_array[np.abs(self.PCexp_array[:, :]) < 1e-10] = 0

        self.nu_array = np.array([0.5*(val + 1/val) for val in self.val_array[:, 0]]).real

        if verbose:
            print('| {} Unitary | {} Complex | {} Real | '.format(num_U, num_C, num_R))
            print('Nus || {:5.10f} | {:5.10f} | {:5.10f} ||'.format(
                self.nu_array[0], self.nu_array[1], self.nu_array[2]))
            print()

    def plot_sol(self, half = False, **args):
        """
        plot_sol - plots solved solution using Simulation.plot_orbit


        Parameters
        ----------
        half: bool
            if True, plots for 2T

        **args:
            arguments passed Simulation.plot_orbit
        
        """

        sc = self.create_sat(s0 = self.S)
        self.propogate(self.T)

        self.plot_orbit(**args)

    def plot_convergence(self, ):
        """
        plot_convergence - plots the covergence of a solution, self.S, self.T

        Parameters
        ----------
        None

        Returns
        ------- 
        None
        
        """

        colors = ['#0051a2', '#97964a', '#ffd44f', '#f4777f', '#93003a']

        iter_array = np.arange(0, len(self.errors), 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex= True)
        Xds = self.Xds
        for i, var in enumerate(self.Xd_vars):
            ax1.plot(iter_array, (Xds[0:-1, i] - Xds[-1, i])/Xds[-1, i], marker = '.', label = var, color = colors[i])

        ax1.set_ylabel('Error $X_d$')
        ax1.legend()

        ax2.plot(iter_array, self.errors, marker = '.', ls = '--', color = 'red')
        ax2.axhline(self.eps, ls = '-', color = 'black')
        ax2.set_yscale('log')

        ax2.set_ylabel(r'Error $|F(X)|$')
        ax2.set_xlabel('Iterations')

        ax2.set_xticks(iter_array)

        ax1.set_title('PyDAA Targeter Report')
        
    def _calc_Xd(self, S0: list, T: float):
        """
        calc_Xd - calculates a design vector from a state and propogation time

        Parameters
        ----------
        S0: 1x6 array
            state vector
        T: float
            time of propogation

        Returns
        -------
        Xd_i: list
            design vector array
        """

        # Set current design variable, subset of state and T
        Xd_i = np.zeros(self.Nd)
        for el, var in enumerate(self.Xd_vars):
            Xd_el = self.var_dict[var]
            if Xd_el == 6:
                Xd_i[el] = T
            else:
                Xd_i[el] = S0[Xd_el]

        return Xd_i

    def _calc_FX(self, Sf: list, T: float):
        """
        _calc_FX - calculates a constraint vector from a state and propogation time

        Parameters
        ----------
        Sf: 1x6 array
            state vector
        T: float
            time of propogation

        Returns
        -------
        FX_i: list
            contraint vector array
        """

        # Set current constraint variable
        FX_i = np.zeros(self.Nc)

        for el, var in enumerate(self.Xc_dict):
            Xc_el = self.var_dict[var]
            if Xc_el == 6:
                FX_i[el] = T - self.Xc_dict[var]
            else:
                FX_i[el] = Sf[Xc_el] - self.Xc_dict[var]

        # if self.psu_arc and self.iter > 0.5:
        #     psu_constraint = (self.Xds[-1] - self.Xd_st).T @ self.null_st - self.step_size
        #     FX_i = np.hstack((FX_i, [psu_constraint]))

        return FX_i

    def _calc_DFX(self, STMf, dS_f):
        """
        _calc_DFX - calculates the Jacobian of the differential corrections algorithm

        Parameters
        ----------
        STMf: 6x6 array
            Final state transition matrix
        dS_f: 1x6 array
            Final state derivative array

        Returns
        -------
        DFX_i: nxm array
            Jacobian of current iteration

        """

        # Create the Jacobian matrix
        DFX_i = np.zeros((self.Nc, self.Nd))
        for i, c_var in enumerate(self.Xc_dict):
            for j, d_var in enumerate(self.Xd_vars):
                el_t = self.var_dict[c_var]
                el_0 = self.var_dict[d_var]
                if el_0 == 6:
                    DFX_i[i, j] = dS_f[el_t]
                else:
                    DFX_i[i, j] = STMf[el_t, el_0]

        # if self.psu_arc and self.iter > 0.5:
        #     psu_jacob = self.null_st
        #     DFX_i = np.vstack((DFX_i, psu_jacob))

        return DFX_i

    def _fill_new_state(self, S_old, T_old, Xd):
        """
        _fill_new_state - creates a new state vector and time of propogation
        from a design vector

        Parameters
        ----------
        S_old: 1x6 array
            old state vector
        T_old: float
            old time of propogation
        Xd: 1xn array
            design vector

        Returns
        -------
        S_new: 1x6 array
            new state from design vector
        T_new: float
            new time of propogation from design vector
            
        """

        S_new = S_old.copy()
        T_new = T_old

        for Xd_el, var in enumerate(self.Xd_vars):
            el = self.var_dict[var]
            if el == 6:
                T_new = Xd[Xd_el]
            else:
                S_new[el] = Xd[Xd_el]

        return S_new, T_new

    def _update_state(self, sat):
        """
        _update_state - takes sc object and solves differential corrections 
        algorithm to produce an updated state and time of propogatio
        
        Parameters
        ----------
        sat: satellite.Satellite object
            propogated sat object 

        Returns
        -------
        S_g: 1x6 array
            guess for state vector 
        T_g: float
            guess for new time of propogation

        **Assigns atributes*

        * self.Xds -- set of design vectors
        * self.error -- current error of targeter
        * self.errors -- adds current error to list
        * self.DFX -- current Jacobian of diff. corrections

        """

        state_func = self.model.get_eom_func()

        states = sat.get_states().T
        taus = sat.get_taus()
        STMs = sat.get_STMs()

        S0, Sf = states[0], states[-1] 
        tau0, tauf = taus[0], taus[-1]

        # final STM and state derivative used in Jacobian calculation
        STMf = STMs[-1]
        dS_f = state_func(tauf, Sf, self.mu, self.e, 
                s = self.s, Js = self.Js, eval_STM = False)

        Xd_i = self._calc_Xd(S0, T = tauf) # Compute design vector
        FX_i = self._calc_FX(Sf, T = tauf) # Compute error vector
        DFX_i = self._calc_DFX(STMf= STMf, dS_f= dS_f) # Compute jacobian
   
        # Calculate updated design variable array
        Xd_ip1 = Xd_i - DFX_i.T @ np.linalg.inv(DFX_i @ DFX_i.T) @ FX_i

        S_g, T_g = self._fill_new_state(S_old= S0, T_old= tauf, Xd = Xd_ip1)

        self.Xds = np.vstack((self.Xds, Xd_ip1))
        self.error = np.linalg.norm(FX_i)
        self.errors.append(self.error)
        self.DFX = DFX_i

        return S_g, T_g


class Continuation(Targeter):
    """
    Continuation - class for implementation of solution continuation procedures

    Parameters
    ----------
    Xd_vars: list of strs
        Vector of design variables, ['x', 'y', 'z', 'vx', 'vy', 'vz', 'T']

        *Example* -- Xd_vars = ['x', 'vy', 'T']

    Xc_dict: dict
        Dictionary of constraint expressions, key -> variable, value -> constriant
        
        Key options: ['x', 'y', 'z', 'vx', 'vy', 'vz']

        *Example* -- {'y' : 0, 'vx' : 0}
    dynamics: str
        name of dynamical model, options: "CR3BP", "ER3BP"
    mu: float
        mass ratio of primaries
    e: float
        eccentricity
    eps:
        precision of targeting algorithm

    Returns
    -------
    None

    """

    def __init__(self, Xd_vars: list, Xc_dict: dict, dynamics: str, 
        mu: float, e: float, eps: float = 1e-10):

        Targeter.__init__(self, Xd_vars, Xc_dict, dynamics, mu, e, eps)

        pass

    def _set_newsol(self, S: list, T: float):
        """
        set_newsol - assigns a new solution

        Parameters
        ----------
        S: 1x6 
            state vector
        T: float
            time of propogation

        Returns
        -------
        None

        *Assigns Attributes*  

        * self.S -- new state arary
        * self.T -- new time of propogation
        
        """

        self.S = S
        self.T = T 
    

    def _value_stop_func(self, S_old: list, T_old: float, S_new: float, T_new: float):
        """
        value_stop - continutation stop function based on value of solution

        Parameters
        ----------
        S_old: 1x6 array
            old state array
        T_old: float
            old time of propogation
        S_new: 1x6 array
            new state array
        T_new: float
            new time of propogation

        Returns
        -------
        continue_bool: bool
            if True, stop condition is not satisfied

        """

        Xd_old = self._calc_Xd(S_old, T_old)[self.stop_var]
        Xd_new = self._calc_Xd(S_new, T_new)[self.stop_var]

        stop_array = np.linspace(Xd_old, Xd_new, 1_000_000)
        stop_calc = np.isclose(stop_array, self.stop_val, rtol = 1e-5)

        if np.sum(stop_calc) == 0:
            continue_bool = True
        else:
            continue_bool = False

        return continue_bool
    
    def _iter_stop_func(self, iterations: int):
        """
        iter_stop_func - continutation stop function based on value of solution

        Parameters
        ----------
        iterations: int
            number of iterations

        Returns
        -------
        continue_bool: bool
            if True, iteration limit not met
        
        """

        if iterations < self.iterstop:
            continue_bool = True 
        else:
            continue_bool = False

        return continue_bool

    def start(self, S_g: list, T_g: float, iter_stop: int = None, stop_dict: dict = None, dXd0_sign: int = +1, 
            step_size: float = 1e-2, type: str = 'psuedo-arc'):
        """
        start - function begins the continuation procedure

        Parameters
        ----------
        S_g: 1x6 array
            initial state vector 
        T_g: float
            intitial time of propogation
        iter_stop: int, optional 
            number of iterations to stop continuation
        stop_dict: dict, optional
            dict that specifies parameter or value to stop continuation
        Xd0_sign: +/-1, optional
            step direction of the first design vector value
        step_size: float, optional
            size of step to take between solutions
        type: str, optional
            type of continuation procedure, Options ['psuedo-arc']
        
        """

        if type == 'psuedo-arc':
            self.psu_arc = True

        if stop_dict != None:
            key = str(list(stop_dict.keys())[0])
            for el in range(len(self.Xd_vars)):

                if str(key) == self.Xd_vars[el]:
                    self.stop_var = el
            # self.stop_var = self.var_dict[key]
            self.stop_val = stop_dict[key]

            continue_bool = self._value_stop_func(S_g, T_g, S_g, T_g)

        if iter_stop != None:
            self.iter = 0
            self.iterstop = iter_stop
            continue_bool = self._iter_stop_func(self.iter)

        if type == 'pseudo-arc':
            cont_func = self.psuedo_step_cont

        self.step_size = step_size

        self.States = []
        self.Ts = []
        self.vecs = []
        self.vals = []
        self.pc_exps = []
        self.nus = []
        self.JCs = []

        while continue_bool:

            self.S, self.T = self.solve(S_g, T_g)
            self.calc_eigenspec(self.S, self.T)
            JC = self.JC_func(tau = 0.0, state = self.S, mu = self.mu, e = self.e)

            self.JCs.append(JC)
            self.States.append(self.S)
            self.Ts.append(self.T)
            self.vecs.append(self.vec_array)
            self.vals.append(self.val_array)
            self.pc_exps.append(self.PCexp_array)
            self.nus.append(self.nu_array)

            self.clear_sats()

            S_g, T_g = self._psuedo_step_cont(dXd0_sign, step_size)

            self.Xd_st = self.Xds[-1]

            if stop_dict != None:
                continue_bool = self._value_stop_func(
                    S_new= S_g, T_new= T_g,
                    S_old = self.S, T_old = self.T)

            if iter_stop != None:
                self.iter += 1
                print(' {:3.2f} %'.format(self.iter/self.iterstop*100))
                continue_bool = self._iter_stop_func(self.iter)

        self.JCs = np.array(self.JCs)
        self.vecs = np.array(self.vecs)
        self.vals = np.array(self.vals)
        self.pc_exps = np.array(self.pc_exps)
        self.nus = np.array(self.nus)


    def _psuedo_step_cont(self, dXd0_sign = +1, step_size = 1e-2, null_el = 0):
        """
        _psuedo_step_cont - implements psuedo-arclength continutation procedure

        Parameters
        ----------
        Xd0_sign: +/-1, optional
            step direction of the first design vector value
        step_size: float, optional
            size of step to take between solutions
        null_el: int, optional
            element of null vector matrix to use


        Returns
        -------
        S_g: 1x6 array
           guess state array 
        T_g: float
            guess time of propogation
        """

        Xd_array = self._calc_Xd(self.S, self.T)
        NA = scipy.linalg.null_space(self.DFX)
        dnull = NA.T[null_el]

        
        if np.isclose(np.sign(dnull[0]), dXd0_sign) != True:
            dnull = -1*dnull
            

        self.NA = scipy.linalg.null_space(self.DFX)
        Xd_ip1 = Xd_array + step_size*dnull

        S_g, T_g = self._fill_new_state(self.S, self.T, Xd_ip1)

        self.S_g = S_g 
        self.T_g = T_g
        self.null_st = dnull

        return S_g, T_g

    def save_cont(self, file_name):
        """
        save_cont - saves the continuation algorithm solutions to an HDF5 file format

        * Attributes saved * 

        * self.States
        * self.Ts
        * self.vals
        * self.vecs
        * self.nus
        * self.PCexp_array

        Parameters
        ----------
        file_name: str
            name of the file saved 
        """
    
        file_path = '{}.hdf5'.format(file_name)

        if os.path.exists(file_path):
            os.remove(file_path)
        # exists = os.path.exists(file_path)
        # i = 0
        # while exists == True:
        #     i += 1
        #     i_str = str(i)
        #     temp_name = '_'.join([file_name, i_str])
        #     temp_path = '{}.hdf5'.format(temp_name)
        #     exists = os.path.exists(temp_path)
        # if i >0:
        #     file_path = temp_path

        f = h5py.File(file_path, "a")
        f.attrs['mu'] = self.mu
        f.attrs['e'] = self.e
        
        save_dict = {
            'IC' : self.States,
            'T' : self.Ts,
            'JC' : self.JCs,
            'm vals' : self.vals,
            'm vecs' : self.vecs,
            'nus' : self.nus,
            'PC exp' : self.PCexp_array

        }
        keys = save_dict.keys()

        for ID in keys:
            f.create_dataset(name = ID, data = save_dict[ID])