
"""
    High-fIdelity multi-dody Dynamics Explorer (HIDE)
    Models.py

    Author: Drew Langford

    Contains restricted dynamcial model state functions and their appropriate
        characteristic values

"""

from families import L2_axial
import numpy as np
from scipy import optimize
import cmath

import matplotlib.pyplot as plt



class Models:
        def __init__(self, dynamics = 'CR3BP'):

            self.dynamics = dynamics

            self.model_dict = {
                'CR3BP' : self.cr3bp_state,
                'ER3BP' : self.er3bp_state,
                'BCR4BP' : self.bcr4bp_state,
                }


            # Earth moon system characteristic values
            mu = 0.0121505856 #M1/(M1 + M2)
            l_st = 384748 # km
            t_st = 375700 # s
            a_s = 149600000/l_st # km -- distance of Sun
            m_s = 1.989e30/(5.97e24 + 0.073e24)

            # Characteristic values of each model
            self.char_dict = {
                'CR3BP' : {
                    'mu' : mu, 
                    'l_st' : l_st,
                    't_st' : 375700, 
                    #'r1' : np.array([-mu, 0 , 0]),
                    #'r2' : np.array([1-mu, 0, 0]),
                    'e' : 0
                    },
                'ER3BP' : {
                    'mu' : mu, 
                    'l_st' : l_st,
                    't_st' : t_st, 
                    'r1' : np.array([-mu, 0 , 0]),
                    'r2' : np.array([1-mu, 0, 0]),
                    'e' : 0.0549
                    },
                'BCR4BP' : {
                    'mu' : mu,
                    'm_s' : m_s,
                    'l_st' : l_st,
                    't_st' : t_st, 
                    'r1' : np.array([-mu, 0 , 0]),
                    'r2' : np.array([1-mu, 0, 0]),
                    'a_s' : a_s,
                    't_sun' : -1/0.9253 #1/13.3712 #characteristic time of sun
                    },
            }

            # Transformation from synodic to inertial frames
            self.transform_dict = {
                'CR3BP' : self.cr3bp_transform,
                'ER3BP' : self.er3bp_transform,
                'BCR4BP' : self.cr3bp_transform, # Not finished?
            }

            # Jacobian matrices
            self.jacobian_dict = {
                'CR3BP' : self.cr3bp_jacobian,
                'ER3BP' : self.er3bp_jacobian,
                'BCR4BP' : self.cr3bp_jacobian, # Not finished
            }
            
            # Primary states
            self.pstates_dict = {
                'CR3BP' : self.cr3bp_pstates,
                'ER3BP' : self.er3bp_pstates,
                'BCR4BP' : self.bcr4bp_pstates
            }

            # Lagrange point states
            self.Lstates_dict = {
                'CR3BP' : self.cr3bp_Lstates,
                'ER3BP' : self.er3bp_Lstates,
                'BCR4BP' : self.cr3bp_Lstates # Not finished
            }

            self.JC_dict = {
                'CR3BP' : self.cr3bp_JC,
            }

            # self.n = 0
            # self.ndot = 0
            # self.xi = 0

            pass 

        def get_state_func(self):

            state_func = self.model_dict[self.dynamics]

            return state_func

        def get_transform_func(self):
            
            transform_func = self.transform_dict[self.dynamics]

            return transform_func

        def get_char_vals(self):

            mu = self.char_dict[self.dynamics]['mu']
            l_st = self.char_dict[self.dynamics]['l_st']
            t_st = self.char_dict[self.dynamics]['t_st']
            #r1_ = self.char_dict[self.dynamics]['r1']
            #r2_ = self.char_dict[self.dynamics]['r2']
            
            return mu, l_st, t_st#, r1_, r2_

        def get_jacobian_func(self):

            jacobian_func = self.jacobian_dict[self.dynamics]

            return jacobian_func

        def get_pstates_func(self):

            pstate_func = self.pstates_dict[self.dynamics]

            return pstate_func

        def get_Lstates_func(self):

            Lstate_func = self.Lstates_dict[self.dynamics]

            return Lstate_func

        def get_JC_func(self):

            JC_func = self.JC_dict[self.dynamics]

            return JC_func

        #############################################################
        ######################### CR3BP #############################
        #############################################################

        def cr3bp_JC(self, tau, s):

            mu = self.char_dict['CR3BP']['mu']
            x, y, z = s[0:3]
            v = np.linalg.norm(s[3:6])
            e_state, m_state = self.cr3bp_pstates(tau)
            r13 = np.linalg.norm(s[0:3] - e_state[0:3])
            r23 = np.linalg.norm(s[0:3] - m_state[0:3])

            U = 0.5*(x**2 + y**2) + (1-mu)/r13 + mu/r23

            JC = 2*U - v**2

            return JC

        def cr3bp_state(self, tau, s, eval_STM = True, p_event = None, epoch = False):
            """ cr3bp_state - CR3BP state function (EOM and STM) for
                numerical integration
            
                    Args:
                        tau (float): non-dim time
                        s (float): non-dim state vector
                        eval_STM (bool): If true, computes STM
                        p_event (tuple) : sets poincare event
                        epoch (bool): if true, calculates derivative of state w/ respect to epoch
            
                    Returns:
                        ds (6x1 array)
                        or 
                        dstate (42x1 array)
            """

            mu = self.char_dict[self.dynamics]['mu']
            e_state, m_state = self.cr3bp_pstates(tau)
            
            # Extract non-dim var for readability in eq. of motion
            x, y, z, xdot, ydot, zdot = s[0:6]

            # Calculate relative position vector form sc to m1/m2
            r13 = np.linalg.norm(s[0:3] - e_state[0:3])
            r23 = np.linalg.norm(s[0:3] - m_state[0:3])

            # CR3BP non-dim equations of motion
            xddot = 2*ydot + x - ((1-mu)/r13**3)*(x + mu) - (mu/r23**3)*(x - (1-mu))
            yddot = -2*xdot + y - ((1-mu)/r13**3)*y - (mu/r23**3)*y
            zddot = - ((1-mu)/r13**3)*z - (mu/r23**3)*z

            ds = np.array([xdot, ydot, zdot, xddot, yddot, zddot])

            if eval_STM:
                # STM Calculations
                Phi_row = s[6:]
                Phi_mat = np.reshape(Phi_row, (6,6))
                A = self.cr3bp_jacobian(s[0:6], tau)
                dPhi_mat = A @ Phi_mat
                dPhi_row = np.hstack(dPhi_mat)

                dstate = np.concatenate((ds, dPhi_row))

                return dstate

            return ds

        def cr3bp_pstates(self, tau):
            """ cr3bp_pstates - returns state vectors of the primaries
                in the synodic frame of the CR3BP

                    Returns:
                        e_state (6x1 array)
                        m_state (6x1 array)
            """
            mu = self.char_dict[self.dynamics]['mu']
            #print(mu)

            r1_ = np.array([   -mu, 0, 0])
            r2_ = np.array([(1-mu), 0, 0])
            v1_ = np.array([0, 0, 0])
            v2_ = np.array([0, 0, 0])

            e_state = np.concatenate((r1_, v1_))
            m_state = np.concatenate((r2_, v2_))

            return e_state, m_state

        def cr3bp_Lstates(self, tau):

            L1 = np.array([0.8369 , 0, 0, 0, 0, 0])
            L2 = np.array([1.1557 , 0, 0, 0, 0, 0])
            L3 = np.array([-1.0051, 0, 0, 0, 0, 0])
            L4 = np.array([0.4878 ,  0.8660, 0, 0, 0, 0])
            L5 = np.array([0.4878 , -0.8660, 0, 0, 0, 0])

            return L1, L2, L3, L4, L5

        def cr3bp_jacobian(self, state, tau):
            """ cr3bp_jacobian - Given a state vector, calculates the jacobian
            matrix 

                Arguments:
                    state (1x6 array): state vector of a sc

                Returns:
                    A (6x6 array): Jacobian Matrix
            
            """

            # SC position in non-dim synodic frame
            x, y, z = state[0:3]
            mu = self.char_dict['CR3BP']['mu']

            # Primary states
            e_state, m_state = self.cr3bp_pstates(tau)

            # Calculate primary distances
            r13 = np.linalg.norm(state[0:3] - e_state[0:3])
            r23 = np.linalg.norm(state[0:3] - m_state[0:3])

            Uxx = 1 - (1 - mu)*(1/r13**3 - (3*(x + mu)**2)/r13**5) - mu*(1/r23**3 - (3*(x - (1-mu))**2)/r23**5)
            Uyy = 1 - (1 - mu)*(1/r13**3 - (3*(y**2))/r13**5) - mu*(1/r23**3 - (3*(y**2))/r23**5)
            Uzz = - (1 - mu)*(1/r13**3 - (3*(y**2))/r13**5) - mu*(1/r23**3 - (3*(y**2))/r23**5)
            Uxy = 3*(1-mu)*y*(x + mu)/r13**5 + 3*mu*y*(x - (1-mu))/r23**5
            Uxz = 3*(1-mu)*z*(x + mu)/r13**5 + 3*mu*z*(x - (1-mu))/r23**5
            Uyz = 3*(1-mu)*y*z/r13**5 + 3*mu*y*z/r23**5

            Uyx = Uxy
            Uzx = Uxz
            Uzy = Uyz

            Udd = np.array([
                [Uxx, Uxy, Uxz],
                [Uyx, Uyy, Uyz],
                [Uzx, Uzy, Uzz]
            ])
            I3x3 = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
            O3x3 = np.zeros_like(I3x3)
            twos3x3 = np.array([
                [ 0, 2, 0],
                [-2, 0, 0],
                [ 0, 0, 0]
            ])
            top = np.concatenate((O3x3, I3x3), axis = 1)
            bottom = np.concatenate((Udd, twos3x3), axis = 1)

            # 36x36 Jacobian
            A = np.concatenate((top, bottom))

            return A

        def cr3bp_transform(self, tau_0, tau):
            """ cr3bp_transform - state transformation matrix from 
                    synodic to inertial reference frame in the CR3BP

                    From Gupta 2020 MS Thesis (Howell Group)
            """
            # Rotation angle
            dtau = tau - tau_0

            IQR = np.array([
                [np.cos(dtau), -np.sin(dtau), 0],
                [np.sin(dtau),  np.cos(dtau), 0],
                [0           ,             0, 1]
            ])
            IQdotR = np.array([
                [-np.sin(dtau), -np.cos(dtau), 0],
                [ np.cos(dtau), -np.sin(dtau), 0],
                [0            ,             0, 0]
            ])
            zero3x3 = np.zeros_like(IQR)

            Qtop = np.concatenate((IQR, zero3x3), axis = 1)
            Qbot = np.concatenate((IQdotR, IQR ), axis = 1)

            # Full state rotation matrix from synodic to inertial frame
            Q = np.concatenate((Qtop, Qbot), axis = 0)

            return Q

        def jacobi_constant(self, state, tau):

            mu = self.char_dict['CR3BP']['mu']
            e_state, m_state = self.cr3bp_pstates(tau)

            x, y, z, vx, vy, vz = state
            r13 = np.linalg.norm(state[0:3] - e_state[0:3])
            r23 = np.linalg.norm(state[0:3] - m_state[0:3])

            U_st = (1-mu)/r13 + mu/r23 + 0.5*(x**2 + y**2)
            vsqr = np.linalg.norm(state[3:])

            JC = 2*U_st - vsqr

            return JC

        #############################################################
        ######################### ER3BP #############################
        #############################################################

        def e_anom(self, tau):
            """ e_anom - finds eccentric anomaly given tau (TA)

            """
            e = self.char_dict[self.dynamics]['e']
            # Find eccentric anamoly
            E0 = tau - e/2 ## good starting point
            E = optimize.newton(self.f_E, E0, args = [tau])

            return E

        def f_E(self, E, tau):
            
            e = self.char_dict[self.dynamics]['e']
            f = E - e*np.sin(E) - tau
            return f

        def e_vals(self, tau):

            e = self.char_dict[self.dynamics]['e']
            # Find eccentric anamoly
            E0 = tau - e/2 ## good starting point
            E = optimize.newton(self.f_E, E0, args = [tau])

            # Calculate eccentric values
            xi = -e*np.cos(E)
            n = np.sqrt(1 - e**2)/(1 - e*np.cos(E))**2
            ndot = (-2*e*np.sqrt(1 - e**2)*np.sin(E))/(1 - e*np.cos(E))**4
            xidot = (e*np.sin(E))/(1-e*np.cos(E))

            return xi, n, ndot, xidot


        def er3bp_state(self, tau, s, eval_STM = True, p_event = None, epoch = False):
            """ er3bp_state - 
                
                    Args:
                        tau (float): ndim time
                        s (nx1 array): state array
                        STM (bool): If true, computes STM
            
                    Returns:
                        ds (6x1 array)
                        or 
                        dstate (42x1 array)
            """
            
            # Calculate eccentric related values
            xi, n, ndot, xidot = self.e_vals(tau)

            # Extract non-dim var for readability in eq. of motion
            x, y, z, xdot, ydot, zdot = s[0:6]

            # Get char values
            mu = self.char_dict[self.dynamics]['mu']

            # Calculate primary distances
            e_state, m_state = self.er3bp_pstates(tau)
            r13 = np.linalg.norm(s[0:3] - e_state[0:3])
            r23 = np.linalg.norm(s[0:3] - m_state[0:3])

            # EOM calculations
            xddot =  2*n*ydot + ndot*y + (n**2)*x - ((1-mu)/r13**3)*(x + mu*(1+xi)) - (mu/r23**3)*(x - (1-mu)*(1+xi))
            yddot = -2*n*xdot - ndot*x + (n**2)*y - ((1-mu)/r13**3)*y - (mu/r23**3)*y
            zddot = -((1-mu)/r13**3)*z - (mu/r23**3)*z

            ds = np.array([xdot, ydot, zdot, xddot, yddot, zddot])

            if eval_STM:
                # STM calculations
                Phi_row = s[6:42]
                Phi_mat = np.reshape(Phi_row, (6,6))
                A = self.er3bp_jacobian(s[0:6], tau)
                dPhi_mat = A @ Phi_mat
                dPhi_row = np.hstack(dPhi_mat)

                dstate = np.concatenate((ds, dPhi_row))

                return dstate

            return ds

        def dfdzeta(self, tau, s, dP, dP_el):
            """ dfdzeta - complex step differentiation for epoch continuity 
                in the ER3BP
            
            """

            # Calculate eccentric related values
            xi, n, ndot, xidot = self.e_vals(tau)

            # Extract non-dim var for readability in eq. of motion
            x, y, z, xdot, ydot, zdot = s[0:6]

            # Get char values
            mu = self.char_dict[self.dynamics]['mu']

            # Calculate primary distances
            e_state, m_state = self.er3bp_pstates(tau)

            e_state = np.array(e_state, dtype = np.complex128)
            m_state = np.array(m_state, dtype = np.complex128)

            # Peturb either primary in one component of their position into the Im axis
            dxi = 1e-7j
            dx = 1e-7
            if dP == 0:
                e_state[dP_el] = e_state[dP_el] + dxi
            elif dP == 1:
                m_state[dP_el] = m_state[dP_el] + dxi
            r13 = np.linalg.norm(np.array(s[0:3], dtype=np.complex128) - e_state[0:3])
            r23 = np.linalg.norm(np.array(s[0:3], dtype=np.complex128) - m_state[0:3])

            # EOM calculations
            xddot =  2*n*ydot + ndot*y + (n**2)*x - ((1-mu)/r13**3)*(x - e_state[0]) - (mu/r23**3)*(x - m_state[0])
            yddot = -2*n*xdot - ndot*x + (n**2)*y - ((1-mu)/r13**3)*(y - e_state[1]) - (mu/r23**3)*(y - m_state[1])
            zddot = -((1-mu)/r13**3)*(z - e_state[2]) - (mu/r23**3)*(z - m_state[2])

            
            ds = np.array([xdot.imag, ydot.imag, zdot.imag, xddot.imag, yddot.imag, zddot.imag])/dx

            return ds

        def er3bp_pstates(self, tau):
            """ er3bp_pstates - returns state vectors of the primaries
                    in the synodic frame of the ER3BP

                    Returns:
                        e_state (6x1 array)
                        m_state (6x1 array)
            """

            mu = self.char_dict[self.dynamics]['mu']
            xi = self.e_vals(tau)[0]
            xidot = self.e_vals(tau)[3]

            r1_ = np.array([-mu*(1 + xi) , 0, 0])
            r2_ = np.array([(1-mu)*(1 + xi), 0, 0])

            v1_ = np.array([-mu*xidot, 0, 0])
            v2_ = np.array([xidot, 0, 0])

            e_state = np.concatenate((r1_, v1_))
            m_state = np.concatenate((r2_, v2_))

            return e_state, m_state

        def er3bp_Lstates(self, tau):

            xi = self.e_vals(tau)[0]

            L1 = np.array([0.8369*(1 + xi) , 0, 0, 0, 0, 0])
            L2 = np.array([1.1557*(1 + xi) , 0, 0, 0, 0, 0])
            L3 = np.array([-1.0051*(1 + xi), 0, 0, 0, 0, 0])
            L4 = np.array([0.4878*(1 + xi) ,  0.8660, 0, 0, 0, 0])
            L5 = np.array([0.4878*(1 + xi) , -0.8660, 0, 0, 0, 0])

            return L1, L2, L3, L4, L5

        def er3bp_jacobian(self, state, tau):
            """ cr3bp_jacobian - Given a state vector, calculates the jacobian
            matrix 

                Arguments:
                    state (1x6 array): state vector of a sc

                Returns:
                    A (6x6 array): Jacobian Matrix
            
            """
            xi, n, ndot, xidot = self.e_vals(tau)

            x, y, z = state[0:3]
            mu = self.char_dict[self.dynamics]['mu']

            r1_ = np.array([-mu*(1 + xi) , 0, 0])
            r2_ = np.array([(1-mu)*(1+xi), 0, 0])
            r13 = np.linalg.norm(state[0:3] - r1_)
            r23 = np.linalg.norm(state[0:3] - r2_)

            # Partials
            Uxx = n**2 - (1-mu)/r13**3 - mu/r23**3 + (3*(1-mu)/r13**5)*(x + mu*(1+xi))**2 + (3*mu/r23**5)*(x - (1-mu)*(1+xi))**2
            Uyy = n**2 - (1-mu)/r13**3 - mu/r23**3 + (3*(1-mu)/r13**5)*y**2               + (3*mu/r23**5)*y**2
            Uzz = 0    - (1-mu)/r13**3 - mu/r23**3 + (3*(1-mu)/r13**5)*z**2               + (3*mu/r23**5)*z**2
            Uxy = (3*(1-mu)/r13**5)*(x + mu*(1+xi))*y + (3*mu/r23**5)*(x - (1-mu)*(1+xi))*y
            Uxz = (3*(1-mu)/r13**5)*(x + mu*(1+xi))*z + (3*mu/r23**5)*(x - (1-mu)*(1+xi))*z
            Uyz = (3*(1-mu)/r13**3)*y*z               + (3*mu/r23**3)*y*z
            Uyx = Uxy
            Uzx = Uxz
            Uzy = Uyz

            Udd = np.array([
                [Uxx, Uxy, Uxz],
                [Uyx, Uyy, Uyz],
                [Uzx, Uzy, Uzz]
            ])
            I3x3 = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
            O3x3 = np.zeros_like(I3x3)
            Rho3x3 = np.array([
                [ 0, 2, 0],
                [-2, 0, 0],
                [ 0, 0, 0]
            ])

            top = np.concatenate((O3x3, I3x3), axis = 1)
            bottom = np.concatenate((Udd + ndot*Rho3x3, n*Rho3x3), axis = 1)

            A = np.concatenate((top, bottom))

            return A

        def er3bp_transform(self, tau_0, tau):
            """ er3bp_transform - state transformation matrix from 
                    synodic to inertial reference frame in the ER3BP

                    THIS IS FAULTY!!
            """
            # Find eccentric values
            e = self.char_dict[self.dynamics]['e']
            E0 = self.e_anom(tau_0)
            E = self.e_anom(tau)
            xi, n, ndot, xidot = self.e_vals(tau)

            dtau = E - E0
            #dtau = tau - tau_0

            alpha = 1
            beta = 1 #1/(1+ e*np.cos(dtau))#1/np.sqrt(1-e**2) #1 #

            cos = np.cos(dtau)   
            sin = np.sin(dtau)         

            IQR = np.array([
                [beta*cos,    -sin, 0],
                [beta*sin,     cos, 0],
                [0       ,      0 , 1]
            ])
            IQdotR = np.array([
                [-sin, -cos, 0],
                [ sin, -cos, 0],
                [0   ,    0, 0]
            ])
            zero3x3 = np.zeros_like(IQR)

            # Stack Matrices
            Qtop= np.concatenate((IQR, zero3x3), axis = 1)
            Qbot = np.concatenate((IQdotR, IQR), axis = 1)

            # 36x36 State rotation matrix
            Q = np.concatenate((Qtop, Qbot), axis = 0)

            return Q

        #############################################################
        ######################## BCR4BP #############################
        #############################################################

        def bcr4bp_state(self, tau, s, eval_STM = False, p_event = None):

            mu = self.char_dict[self.dynamics]['mu']
            m_s = self.char_dict[self.dynamics]['m_s'] 
            t_sun = self.char_dict[self.dynamics]['t_sun']
            a_s = self.char_dict[self.dynamics]['a_s']
            e_state, m_state, s_state = self.bcr4bp_pstates(tau)

            # Extract non-dim var for readability in eq. of motion
            x, y, z, xdot, ydot, zdot = s[0:6]

            # Calculate relative position vector form sc to m1/m2
            r14 = np.linalg.norm(s[0:3] - e_state[0:3])
            r24 = np.linalg.norm(s[0:3] - m_state[0:3])
            r34 = np.linalg.norm(s[0:3] - s_state[0:3])

            # CR3BP non-dim equations of motion
            xddot = 2*ydot + x - ((1-mu)/r14**3)*(x + mu) - (mu/r24**3)*(x - (1-mu)) - \
                (m_s/r34**3)*(x - a_s*np.cos(tau/t_sun)) - (m_s/a_s**2)*np.cos(tau/t_sun)

            yddot = -2*xdot + y - ((1-mu)/r14**3)*y - (mu/r24**3)*y - \
                (m_s/r34**3)*(x - a_s*np.sin(tau/t_sun)) - (m_s/a_s**2)*np.sin(tau/t_sun)

            zddot = - ((1-mu)/r14**3)*z - (mu/r24**3)*z - (m_s/r34**3)*z

            ds = np.array([xdot, ydot, zdot, xddot, yddot, zddot])

            return ds

        def bcr4bp_pstates(self, tau):

            mu = self.char_dict[self.dynamics]['mu']
            a_s = self.char_dict[self.dynamics]['a_s']
            t_sun = self.char_dict[self.dynamics]['t_sun']

            r1_ = np.array([   -mu, 0, 0])
            r2_ = np.array([(1-mu), 0, 0])
            r3_ = a_s*np.array([np.cos(tau/t_sun), np.sin(tau/t_sun), 0])
            v1_ = np.array([0, 0, 0])
            v2_ = np.array([0, 0, 0])
            v3_ = a_s/t_sun*np.array([-np.sin(tau/t_sun), np.cos(tau/t_sun), 0]) # maybe not correct..

            e_state = np.concatenate((r1_, v1_))
            m_state = np.concatenate((r2_, v2_))
            s_state = np.concatenate((r3_, v3_))

            return e_state, m_state, s_state

if __name__ == '__main__':


    pass