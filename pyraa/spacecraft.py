"""
    High-fidelity multi-body Dynamics Explorer (HiDE)
    spacecraft.py

    Author: Drew Langford

"""
import numpy as np

class Spacecraft:

        def __init__(self, s0, tau0, monte = False):
            """ __init__ - initializes spacecraft object and its properties
                 
                 Args:
                    s0 (1x6 array): initial non-dim state of the sc
                    tau0 (float): initial non-dim time of the sc
                    monte (bool): monte_sc flag

                Returns:
                    None
            
            """
            # Set up initial state transition matrix
            Phi0 = np.identity(6)
            Phi_row0 = np.hstack(Phi0)
            
            self.STM = Phi_row0
            self.STMs = np.array([Phi_row0])
            self.s = s0
            self.states = np.array([s0])
            self.tau = tau0
            self.taus = np.array([tau0])

            self.e = 0

            self.monte = monte

            pass 

        def get_state(self,):

            return self.s[0:6]

        def get_states(self,):

            return self.states.T[0:6]
        
        def get_tau(self):

            return self.tau

        def get_taus(self):
            
            return self.taus

        def get_STM(self):

            STM = np.reshape(self.STM, (1, 6, 6))[0]

            return STM

        def get_STMs(self):

            STMs = np.reshape(self.STMs, (len(self.STMs), 6, 6))

            return STMs

        def set_state(self, s_new):

            self.s = s_new.T[-1]
            self.states = np.concatenate((self.states, s_new.T), axis = 0)

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