"""
    satellite module ``satellite.py`` 
    
    This module contains ``pyraa.satellite.Satellite`` object class 

    :Authors: Drew Langford

    :Last Edit: 
        Langford, 06/2022

"""
# Standard imports

# Third Party imports
import numpy as np

# PyRAA imports
import pyraa.dyn_sys_tools as dsys_utils
from pyraa.models import Models

class Satellite(object):
    """ 
    Satellite - serves as data class object for trajectory propogation
            
    Parameters
    ----------
    S0: 1x6 array
        initial state vector
    tau0: float, optional
        system epoch at intial state
    color: str or rbg code, optional
        assigns color for plotting
    lw: float
        line-width used in plotting
    ls: str
        line-style used in plotting
    alpha: float
        opacity used in plotting

    Returns
    -------
        None
    
    """
    def __init__(self, S0, tau0: float, color = None, lw: float = 1, 
        ls = '-', alpha: float = 1):

        model = Models('CR3BP') #Doesn't matter what model
        transform_func = model.get_transform_func()
        Q0 = transform_func(tau = np.float(tau0), e = np.float(0))

        self.state = S0 
        self.state_inert = Q0 @ S0
        self.tau = tau0 

        Phi0 = np.identity(6)  # Set up initial state transition matrix
        Phi_row0 = np.hstack(Phi0)
        self.STM = Phi_row0

        self.states = np.array([S0])
        self.states_inert = np.array([self.state_inert])
        self.taus = np.array([tau0])
        self.STMs = np.array([Phi_row0])
        
        self.e = 0 ####?????

        # Init plotting properties
        self.color = color
        self.lw = lw
        self.ls = ls
        self.alpha = 1
        self.marker = '.'
        pass 

    def get_state(self,):
        """
        get_state - returns current state of sat

        Returns
        -------
        state: 
            current state of sat
        """

        state = self.state[0:6]

        return state

    def get_states(self,):
        """
        get_states - returns all saved states of sat

        Returns
        -------
        states: 
            saved states of sat
        """

        states = self.states.T[0:6]

        return states
    
    def get_tau(self):
        """
        get_tau - returns current time

        Returns
        -------
        tau: float
            current time of sat
        """

        return self.tau

    def get_taus(self):
        """
        get_taus - returns all saved times of sat

        Returns
        -------
        taus: array of floats
            saved times of sat
        """
        
        return self.taus

    def get_STM(self):

        STM = np.reshape(self.STM, (1, 6, 6))[0]

        return STM

    def get_STMs(self):

        STMs = np.reshape(self.STMs, (len(self.STMs), 6, 6))

        return STMs

    def set_state(self, s_new):

        self.state = s_new.T[-1]
        self.states = np.concatenate((self.states, s_new.T), axis = 0)

        pass

    def set_inert_states(self, s_new):

        self.state_inert = s_new.T[-1]
        self.states_inert = np.concatenate((self.states_inert, s_new.T), axis = 0)

        pass

    def set_tau(self, tau_new):
        
        self.tau = tau_new[-1]
        self.taus = np.concatenate((self.taus, tau_new))

        pass 

    def set_STM(self, STM_new):

        self.STM = STM_new.T[-1]
        self.STMs = np.concatenate((self.STMs, STM_new.T), axis = 0)

        pass

    def set_events(self, events):

        self.events = events 
    
    def set_etaus(self, etaus):

        self.etaus = etaus

    def set_JCs(self, JCs):

        self.JCs = JCs

    def set_dxdtaus(self, dxdtaus):

        self.dxdtaus= dxdtaus

    def get_dxdtaus(self):

        return self.dxdtaus

    def get_events(self):
        
        return self.events

    def get_etaus(self,):

        return self.etaus

    def get_JCs(self):

        return self.JCs

    def get_e(self):

        return self.e

    def set_e(self, e):

        self.e = e

    def set_FTLEs(self, FTLEs):
        self.FTLEs = FTLEs

    def get_FTLEs(self):
        return self.FTLEs

