"""
    dynamical models module ``models.py`` 
    
    This module contains functions to perform calculations in
    restricted multi-body dynamical models

    .. tip::
        All functions for a given model can be called through the ``models.Model`` 
        class

    .. attention::
        Performance critical calculations are now performed with ``numba``!

    :Authors: 
        Drew Langford

    :Last Edit: 
        Langford, 06/2022

"""

import json
import time

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy import optimize as sci_op
from scipy import special as sci_sp


def create_system(name, mu, e = None, m1 = None, m2 = None, a12 = None, 
                    t12 = None, m3 = None, a3 = None, t3 = None):

    sys_dict = {
        'Name' : name,
        'mu' : mu,
        'e' : e,
        'm1' : m1,
        'm2' : m2,
        'm3' : m3,
        'a12' : a12,
        'a3' : a3,
        't12' : t12,
        't3' : t3
    }

    with open('systems.json', 'a') as outfile:
        json.dump(sys_dict, outfile)



systems = {
    'EarthMoon' : {
        'm1' : 5.97e24,
        'm2' : 0.073e24,
        'mu ' : 0.0121505856,
        'e' : 0.0549,
        'l_st' : 384748,
        't_st' : 375700,
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
    }

}

class Models:
    """
    Models - class object to call functions related to a specified 
    restricted dynamical model

    .. Note:

        Hello, there

    Parameters
    -----------
    dynamics: str, optional
        sets restricted dynamical model of choice. 
        Options include: 'CR3BP', 'ER3BP'.
        'BCR4BP' coming soon!

    Returns
    -------
    None
    """
    def __init__(self, dynamics = 'CR3BP'):

        self.dynamics = dynamics

        # Equation of motion models
        self.eom_dict = {
            'CR3BP' : cr3bp_eom,
            'ER3BP' : er3bp_eom,
            #'BCR4BP' : self.bcr4bp_state,
            }

        # Jacobian matrices for models
        self.jacobian_dict = {
            'CR3BP' : cr3bp_jacobian,
            'ER3BP' : er3bp_jacobian,
            # 'BCR4BP' : self.cr3bp_jacobian, # Not finished
            }

        # Jacobi Constant evaluation
        self.JC_dict = {
            'CR3BP' : cr3bp_JC,
            'ER3BP' : er3bp_JC,
            }
        
        # Primary states of models
        self.pstates_dict = {
            'CR3BP' : cr3bp_pstates,
            'ER3BP' : er3bp_pstates,
            # 'BCR4BP' : self.bcr4bp_pstates
            }

        # Lagrange point states
        self.Lstates_dict = {
            'CR3BP' : cr3bp_Lstates,
            'ER3BP' : er3bp_Lstates,
            # 'BCR4BP' : self.cr3bp_Lstates # Not finished
            }

        self.transform_dict = {
            'CR3BP' : cr3bp_transform,
            'ER3BP' : er3bp_transform
        }

        pass 

    def get_eom_func(self):
        """
        get_eom_func - returns function for model's equations of motion

        The eom_func may be used as the RHS in numerical ode integration


        Parameters
        ----------
        None

        Returns
        -------
        eom_func: function
            fucntion evaluates the dynamic's equations of motion 
        
        """

        eom_func = self.eom_dict[self.dynamics]

        return eom_func


    def get_jacobian_func(self):
        """
        get_jacobian_func - returns function for model's Jacobian matrix

        The Jacobian, df/dx, is implemented while integrating the STM

        Parameters
        ----------
        None

        Returns
        -------
        jacobian_func: function
            fucntion evaluates the dynamical models Jacobian matrix
        
        """

        jacobian_func = self.jacobian_dict[self.dynamics]

        return jacobian_func

    def get_JC_func(self):
        """
        get_JC_func - returns function for calculating Jacobi Constant

        ** Note, the Jacobi Constant in **not** invariant in 
        non-autonomous models, e.g. 'ER3BP'

        Parameters
        ----------
        None

        Returns
        -------
        jacobian_func: function
            function evaluates the Jacobi constant
        
        """

        JC_func = self.JC_dict[self.dynamics]

        return JC_func

    def get_pstates_func(self):
        """
        get_pstates_func - returns function for calculating primary states

        Parameters
        ----------
        None

        Returns
        -------
        pstates_func: function
            function evaluates the dynamical model's primary state vectors
        
        """

        pstate_func = self.pstates_dict[self.dynamics]

        return pstate_func

    def get_Lstates_func(self):
        """
        get_Lstates_func - returns function for calculating Lagrange point states

        Parameters
        ----------
        None

        Returns
        -------
        Lstates_func: function
            function evaluates the dynamical model's Lagrange point state vectors
        
        """

        Lstate_func = self.Lstates_dict[self.dynamics]

        return Lstate_func

    def get_transform_func(self):
        """
        get_transform_func - returns function for calculating 
        synodic to inertial transformation matrix

        Parameters
        ----------
        None

        Returns
        -------
        transform_func: function
            function evaluates the dynamical model's Lagrange point state vectors
        
        """

        transform_func = self.transform_dict[self.dynamics]

        return transform_func


##########################################################
##################### General Methods ####################
##########################################################
"""
Methods used by one or more models which have general
purpose use

Methods
-------
LPx_func - zero function for x value of collinear eq. points
E_anom_func - zero function for Eccen. Anomaly
calc_E_anom - calculates Eccen. Anomaly using Newton's method
calc_f_Anom - calculates true anomaly from E, e
dfdzeta - complex step differentiation for epoch continuity 
        in the ER3BP !!!! IN DEVELOPMENT
"""

@nb.njit()
def lpx_func(x: float, mu: float, A: int, B: int):
    """
    calc_LPx - zero function for x value of collinear eq. points

    Function called in root-finding algorithm for x-position of 
    collinear Lagrange points

    *Accelerated* with ``numba``
    
    Parameters
    ----------
    x: float
        x-position 
    mu: float
        mass ratio of system
    A: int
        sign of p1 force term
    B: int
        sign of p2 force term

    Returns
    -------
    f_lpx: float
        function value
    """

    term1 = x
    term2 = (1-mu)/np.power(x+mu, 2)
    term3 = (mu)/np.power(x-1+mu, 2)
    
    f_lpx = term1 + A*term2 + B*term3
    
    return f_lpx 

@nb.njit
def ecc_anom_func(E: float, M: float, e: float):
    """
    ecc_anom_func - zero function for eccentric anomaly

    Function is used with root finding algorithm to calculate eccentric anomaly

    *Accelerated* with ``numba``

    Parameters
    ----------
    E: float
        eccentric anomaly
    M: float
        mean anomaly
    e: float
        eccentricity

    Returns
    -------
    f_e: float
        function value, E - e*sin(E) - M
    """

    f = E - e*np.sin(E) - M
    
    return f
    
def calc_ecc_anom(M: float, e: float):
    """
    calc_E_anom - calculates Eccen. Anomaly using Newton's method

    Function calls E_anom_func to find root with ``scipy.optimize.newton``

    Parameters
    ----------
    M: float
        mean anomaly
    e: float
        eccentricity

    Returns
    -------
    E: float
        eccentric anomaly
    """
    
    E0 = M - e/2
    E = sci_op.newton(ecc_anom_func, E0, args = [M, e], maxiter = 100)
    
    return E

@nb.njit(cache = True)
def calc_E_series(M: float, e: float, s, Js):
    """
    calc_E_series - calculates eccentric anomaly based on series expansion

    *Accelerated* with ``numba``

    Uses Bessel function series expansion found in:
    
    * `Murray & Durmott 1999 <https://doi.org/10.1017/CBO9781139174817>`_
    * `Philcox, Goodman, Slepian 2021 <https://arxiv.org/pdf/2103.15829.pdf>`_

    Parameters
    ----------
    M: float
        mean anomaly
    e: float
        eccentricity 
    s: 1xN array
        series expansion indicies [1, N]
    Js: 1xN array
        precomputed Bessel function of indicies [1, N]

    Returns
    -------
    E: float
        eccentric anomaly
    """

    E = M + 2*np.sum(Js*np.sin(s*M)/s)

    return E

@nb.njit
def calc_f_anom(E: float, e: float):
    """
    calc_f_Anom - calculates true anomaly from E, e

    *Accelerated* with ``numba``

    Parameters
    ----------
    E: float
        eccentric anomaly
    e: float
        eccentricity of system

    Returns
    -------
    f: float
        true anomaly
    """

    f = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    
    return f


# def dfdzeta(self, tau, s, dP, dP_el):
#     """ 
#     dfdzeta - complex step differentiation for epoch continuity 
#         in the ER3BP
    
#     """

#     # Calculate eccentric related values
#     xi, n, ndot, xidot = self.e_vals(tau)

#     # Extract non-dim var for readability in eq. of motion
#     x, y, z, xdot, ydot, zdot = s[0:6]

#     # Get char values
#     mu = self.char_dict[self.dynamics]['mu']

#     # Calculate primary distances
#     e_state, m_state = self.er3bp_pstates(tau)

#     e_state = np.array(e_state, dtype = np.complex128)
#     m_state = np.array(m_state, dtype = np.complex128)

#     # Peturb either primary in one component of their position into the Im axis
#     dxi = 1e-7j
#     dx = 1e-7
#     if dP == 0:
#         e_state[dP_el] = e_state[dP_el] + dxi
#     elif dP == 1:
#         m_state[dP_el] = m_state[dP_el] + dxi
#     r13 = np.linalg.norm(np.array(s[0:3], dtype=np.complex128) - e_state[0:3])
#     r23 = np.linalg.norm(np.array(s[0:3], dtype=np.complex128) - m_state[0:3])

#     # EOM calculations
#     xddot =  2*n*ydot + ndot*y + (n**2)*x - ((1-mu)/r13**3)*(x - e_state[0]) - (mu/r23**3)*(x - m_state[0])
#     yddot = -2*n*xdot - ndot*x + (n**2)*y - ((1-mu)/r13**3)*(y - e_state[1]) - (mu/r23**3)*(y - m_state[1])
#     zddot = -((1-mu)/r13**3)*(z - e_state[2]) - (mu/r23**3)*(z - m_state[2])

#     ds = np.array([xdot.imag, ydot.imag, zdot.imag, xddot.imag, yddot.imag, zddot.imag])/dx

#     return ds


##########################################################
##################### CR3BP Methods ######################
##########################################################
"""
Methods used for CR3BP dynamical model

Methods
-------

cr3bp_eom - CR3BP state function (EOM and STM) for numerical integration
cr3bp_jacobian - calculates the jacobian matrix of cr3bp eom 
cr3bp_JC - calculate Jacobi constant in cr3bp model
cr3bp_pstates - returns state vectors of the primaries in the cr3bp
cr3bp_Lstates - calculates and returns array of Lagrange points in the cr3bp
cr3bp_transform - computes transformation matrix from synodic to inertial frame

"""

@nb.njit(cache = True)
def cr3bp_eom(tau: float, state: list, mu: float, e: float = 0, s: list = None, Js: list = None,
                eval_STM: bool = True, p_event: tuple = None, epoch: bool = False):
    """ 
    cr3bp_eom - CR3BP state function (EOM and STM) for numerical integration

    *Accelerated* with ``nb.njit(cache = True)``
    
    Parameters
    ----------
    tau: float
        non-dim time
    state: 1x6 array
        non-dim state vector
    mu: float
        mass ratio of system
    e: float, optional
        eccentricity of system
    s: 1xN array, optional
        pre-computed indicies for eccentric anomaly series
        **NOT** used in CR3BP
    Js: 1xN array, optional
        pre-computed Bessel funcs for eccentric anomaly series 
        **NOT** used in CR3BP
    eval_STM: bool, optional
        if True, computes STM for integration
    p_event: tuple, optional
        sets poincare event
    epoch: bool, optional
        if True, calculates derivative of state w/ respect to epoch
    
    Returns
    -------
    dstate: 1x6 array **or** 1x42 array, if ``eval_STM = True``
        derivative of state array
    """

    p1_state, p2_state = cr3bp_pstates(tau, mu)
    
    # Extract non-dim var for readability in eq. of motion
    x, y, z, xdot, ydot, zdot = state[0:6]

    # Calculate relative position vector form sc to m1/m2
    r13 = np.linalg.norm(state[0:3] - p1_state[0:3])
    r23 = np.linalg.norm(state[0:3] - p2_state[0:3])

    # CR3BP non-dim equations of motion
    xddot = 2*ydot + x - ((1-mu)/r13**3)*(x + mu) - (mu/r23**3)*(x - (1-mu))
    yddot = -2*xdot + y - ((1-mu)/r13**3)*y - (mu/r23**3)*y
    zddot = - ((1-mu)/r13**3)*z - (mu/r23**3)*z

    dstate = np.array([xdot, ydot, zdot, xddot, yddot, zddot])

    if eval_STM: # STM Calculations
        Phi_row = state[6:].copy()
        Phi_mat = Phi_row.reshape((6,6))
        A = cr3bp_jacobian(tau, state[0:6], mu)
        dPhi_mat = A @ Phi_mat
        dPhi_row = dPhi_mat.flatten()

        dstate_STM = np.zeros(42)
        dstate_STM[0: 6] = dstate
        dstate_STM[6: ] = dPhi_row

        return dstate_STM

    return dstate

@nb.njit(cache = True)
def cr3bp_jacobian(tau: float, state: list, mu: float):
    """ 
    cr3bp_jacobian - calculates the Jacobian matrix of CR3BP EOM

    *Accelerated* with ``nb.njit(cache = True)``

    Parameters
    ----------
    tau: float
        non-dim time
    state: 1x6 array
        state vector 
    mu: float
        mass ratio of system

    Returns
    -------
    A: 6x6 array
        evaluated Jacobian matrix
    """

    # SC position in non-dim synodic frame
    x, y, z = state[0:3]

    # Primary states
    p1_state, p2_state = cr3bp_pstates(tau, mu)

    # Calculate primary distances
    r13 = np.linalg.norm(state[0:3] - p1_state[0:3])
    r23 = np.linalg.norm(state[0:3] - p2_state[0:3])

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

    # 6x6 Jacobian
    A = np.concatenate((top, bottom))

    return A

@nb.njit(cache = True)
def cr3bp_JC(tau: float, state: list, mu: float, e: float = 0):
    """
    cr3bp_JC - calculate Jacobi constant in cr3bp model

    *Accelerated* with ``nb.njit(cache = True)``

    Parameters
    ----------
    tau: float
        non-dim time
    state: 1x6 array
        state of sc
    mu: float
        mass ratio of system
    e: float, optional
        eccentricity of system, does **not** effect calculation in 
        CR3BP

    Returns
    -------
    JC: float
        jacobi constant
    """

    x, y, z = state[0:3]
    v = np.linalg.norm(state[3:6])
    
    p1_state, p2_state = cr3bp_pstates(tau, mu)
    r13 = np.linalg.norm(state[0:3] - p1_state[0:3])
    r23 = np.linalg.norm(state[0:3] - p2_state[0:3])

    U = 0.5*(x**2 + y**2) + (1-mu)/r13 + mu/r23
    
    JC = 2*U - v**2

    return JC

@nb.njit(cache = True)
def cr3bp_pstates(tau: float, mu: float, e: float = 0):
    """ 
    cr3bp_pstates - returns state vectors of the primaries in the synodic 
    frame of the CR3BP

    *Accelerated* with ``nb.njit(cache = True)``

    Parameters 
    ----------
    tau: float
        non-dim time
    mu: float 
        mass ratio of system
    e: float
        eccentricity of system

    Returns
    -------
    p1_state : 1x6 array
        state of p1
    p2_state : 1x6 array
        state of p2 
    """

    r1_ = np.array([   -mu, 0, 0])
    r2_ = np.array([(1-mu), 0, 0])
    v1_ = np.array([0, 0, 0])
    v2_ = np.array([0, 0, 0])

    p1_state = np.concatenate((r1_, v1_))
    p2_state = np.concatenate((r2_, v2_))

    return p1_state, p2_state


@nb.jit(cache = True)
def cr3bp_Lstates(tau: float, mu: float, e: float):
    """
    cr3bp_Lstates - calculates and returns array of Lagrange points in the cr3bp

    In the CR3BP there are 3 collinear and 2 equilateral equilibrium points.
    This function computes the state vector of L1-L5. 

    **Source of Implementation**

    `Short 2010 Master's Thesis Purdue University 
    <https://engineering.purdue.edu/people/kathleen.howell.1/Publications/masters/2010_Short.pdf>`_

    Parameters
    ----------
    tau: float
        non-dim time
    mu: float
        mass ratio of system
    e: float
        eccentricity of system

    Returns
    -------
    L_states: 5x6 array
        states of Lpoints ordered 1-5
    """
    
    # Sets signs of eq. function
    A1, B1 = -1, +1
    A2, B2 = -1, -1
    A3, B3 = +1, +1

    xL1 = sci_op.bisect(f = lpx_func, a = 0, b = 1 - mu - 1e-5, args = (mu, A1, B1))
    xL2 = sci_op.bisect(f = lpx_func, a = 1 - mu + 1e-5, b = 3, args = (mu, A2, B2))
    xL3 = sci_op.bisect(f = lpx_func, a = -3, b = - mu - 1e-5, args = (mu, A3, B3))
    
    L1 = np.array([xL1, 0, 0, 0 , 0, 0])
    L2 = np.array([xL2, 0, 0, 0 , 0, 0])
    L3 = np.array([xL3, 0, 0, 0 , 0, 0])
    L4 = np.array([1/2 - mu, + np.sqrt(3)/2, 0, 0 , 0, 0])
    L5 = np.array([1/2 - mu, - np.sqrt(3)/2, 0, 0 , 0, 0])
    
    L_states = np.vstack((L1, L2, L3, L4, L5))
    
    return L_states

@nb.njit
def cr3bp_transform(tau: float, e: float):
    """
    cr3bp_transform - computes transformation matrix from synodic to inertial frame

    General rotation matrix aout the z-axis 

    *Accelerated* with ``nb.njit(cache = True)``

    Parameters
    ----------
    tau: float
        epoch of transformation, this is equivalent to the true anomaly of the binary
        bounded [0, 2pi)
    e: float
        eccentricity of system

    Returns
    -------
    Q: 6x6 array
        evaluated rotation matrix Q(0, tau)
    """

    cos = np.cos(tau)
    sin = np.sin(tau)

    q = np.array([[cos, -sin, 0.0], [sin,  cos, 0.0],
        [0.0,    0.0, 1.0]
    ], dtype = np.float64)

    Q = 0

    qdot = np.array([
        [-sin, -cos, 0.0],
        [cos,  -sin, 0.0],
        [0.0,    0.0, 1.0]
    ], dtype = np.float64)
    
    O3x3 = np.zeros((3, 3))

    Qtop = np.concatenate((q, O3x3), axis = 1)
    Qbot = np.concatenate((qdot, q), axis = 1)

    Q = np.vstack((Qtop, Qbot))

    return Q



##########################################################
##################### ER3BP Methods ######################
##########################################################
"""
Methods used for ER3BP dynamical model

Methods
-------
er3bp_eom - calculates state derivative using er3bp eom
er3bp_jacobian - calculates jacobian matrix of er3bp EOM
er3bp_JC - calculate Jacobi constant in er3bp model
er3bp_pstates - calculates state of primaries in er3bp
er3bp_Lstates - calculates and returns array of Lagrange points in the er3bp
er3bp_transform - computes transformation matrix from synodic to inertial frame

"""

@nb.njit(cache = True)
def er3bp_eom(tau: float, state: list, mu: float, e: float, s: list, Js: list, eval_STM: bool = False, 
                p_event: tuple = None, epoch: bool = False):
    """
    er3bp_eom - calculates state derivative using ER3BP EOM

    *Accelerated* with ``nb.njit(cache = True)``

    Source: 
    `Hiday 1992 Purdue Univerity Dissertation 
    <https://engineering.purdue.edu/people/kathleen.howell.1/Publications/dissertations/1992_Hiday.pdf>`_

    .. Note:: 
        ER3BP is a non-autonomous dynamical system, i.e. its EOM are a function 
        of state variables **and** time
    
    Parameters
    ----------
    tau: float
        non-dim time
    state: 1x6 array
        non-dim state vector
    mu: float
        mass ratio of system
    e: float
        eccentricity of system
    s: 1xN array of ints
        pre-computed integer array for E anom
    Js: 1xN array 
        pre-computed Bessel func of first kind for E anom
    eval_STM: bool 
        if true, computes STM for integration
    p_event: tuple 
        sets poincare event
    epoch: bool
        if true, calculates derivative of state w/ respect to epoch
    
    Returns
    -------
    dstate: 1x6 array **or** 1x42 array, if ``eval_STM = True``
        derivative of state array
    """

    ## Calculates Eccen. Anomaly using series expansion
    E = calc_E_series(tau, e, s, Js)

    # Calculate primary distances
    p1_state, p2_state = er3bp_pstates(E, mu, e)

    # Extract state variables
    x, y, z = state[0: 3]
    xdot, ydot, zdot = state[3:6]
    r_ = state[0:3]
    v_ = state[3:]

    x1 = p1_state[0]
    x2 = p2_state[0]
    r13 = np.linalg.norm(state[0:3] - p1_state[0:3])
    r23 = np.linalg.norm(state[0:3] - p2_state[0:3])

    # compute inst. frame rotation and change in rotation
    n = np.sqrt(1-e**2)/np.power(1-e*np.cos(E), 2)
    ndot = -2*e*np.sqrt(1-e**2)*np.sin(E)/np.power(1-e*np.cos(E), 4)

    # precompute distance cubed
    r13_cu = np.power(r13, 3)
    r23_cu = np.power(r23, 3)

    # EOM calculations
    xddot =  2*n*ydot + ndot*y + (n**2)*x - ((1-mu)/r13_cu)*(x - x1) - (mu/r23_cu)*(x - x2)
    yddot = -2*n*xdot - ndot*x + (n**2)*y - ((1-mu)/r13_cu)*y - (mu/r23_cu)*y
    zddot = -((1-mu)/r13_cu)*z - (mu/r23_cu)*z
    
    dstate = np.array([xdot, ydot, zdot, xddot, yddot, zddot])

    if eval_STM:
        # STM calculations
        Phi_row = state[6:42].copy()
        Phi_mat = np.reshape(Phi_row, (6, 6))
        A = er3bp_jacobian(E, state[0:6], p1_state[0:3], p2_state[0:3], mu, e)
        dPhi_mat = A @ Phi_mat
        dPhi_row = dPhi_mat.flatten()
        dstate_STM = np.zeros(42)
        dstate_STM[0: 6] = dstate
        dstate_STM[6: ] = dPhi_row

        return dstate_STM

    return dstate


@nb.njit(cache = True)
def er3bp_jacobian(E: float, state: list, r1_: list, r2_: list, mu: float, e: float):
    """
    er3bp_jacobian - calculates Jacobian matrix of ER3BP EOM

    *Accelerated* with ``nb.njit(cache = True)``

    Source: 
    `Hiday 1992 Purdue Univerity Dissertation 
    <https://engineering.purdue.edu/people/kathleen.howell.1/Publications/dissertations/1992_Hiday.pdf>`_

    
    Parameters
    ----------
    E: float
        eccentric anomaly of binary
    state: 1x6 array 
        state vector
    r1_: 1x3 array
        position of p1
    r2_: 1x3 array
        position of p2
    mu: float
        mass ratio of system
    e: float
        eccentricity of system

    Returns
    -------
    A: 6x6 array
        evaluated Jacobian matrix
    """
    
    r1 = r1_[0]
    r2 = r1_[0]
    r13 = np.linalg.norm(state[0:3] - r1_)
    r23 = np.linalg.norm(state[0:3] - r2_)
    
    n = np.sqrt(1-e**2)/np.power(1-e*np.cos(E), 2)
    ndot = -2*e*np.sqrt(1-e**2)*np.sin(E)/np.power(1-e*np.cos(E), 4)
    x, y, z = state[0:3]

    # Partials
    Uxx = n**2 - (1-mu)/r13**3 - mu/r23**3 + (3*(1-mu)/r13**5)*(x - r1)**2 + (3*mu/r23**5)*(x - r2)**2
    Uyy = n**2 - (1-mu)/r13**3 - mu/r23**3 + (3*(1-mu)/r13**5)*y**2               + (3*mu/r23**5)*y**2
    Uzz = 0    - (1-mu)/r13**3 - mu/r23**3 + (3*(1-mu)/r13**5)*z**2               + (3*mu/r23**5)*z**2
    Uxy = (3*(1-mu)/r13**5)*(x - r1)*y + (3*mu/r23**5)*(x - r2)*y
    Uxz = (3*(1-mu)/r13**5)*(x - r1)*z + (3*mu/r23**5)*(x - r2)*z
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

@nb.njit(cache = True)
def er3bp_pstates(E: float, mu: float, e: float):
    """
    er3bp_pstates - calculates state of primaries in ER3BP

    *Accelerated* with ``nb.njit(cache = True)``

    Source

    `Hiday 1992 Purdue Univerity Dissertation 
    <https://engineering.purdue.edu/people/kathleen.howell.1/Publications/dissertations/1992_Hiday.pdf>`_


    Parameters
    ----------
    E: float
        eccentric anomaly
    mu: float
        mass ratio of system
    e: float
        eccentricity

    Returns
    --------
    p1_state: 1x6 array
        state of p1
    p2_state: 1x6 array
        state of p2 
    """

    # distance from barycenter from orbit eq
    x1 = -mu*(1-e*np.cos(E)) 
    x2 = (1-mu)*(1-e*np.cos(E)) 
    r12 = np.abs(x2-x1)
    
    Edot = 1/(1-e*np.cos(E))
    
    xdot1 = -mu*e*np.sin(E)*Edot
    xdot2 = (1-mu)*e*np.sin(E)*Edot
    
    p1_state = np.array([x1, 0, 0, xdot1, 0, 0])
    p2_state = np.array([x2, 0, 0, xdot2, 0, 0])
    
    return p1_state, p2_state

def er3bp_Lstates(E: float, mu: float, e:float):
    """
    er3bp_Lstates - calculates and returns array of Lagrange points in the ER3BP

    .. Note::
        In the ER3BP there are 3 instantaneous collinear equilibrium points and 2 static
        equilateral equilibrium points

    Sources

    `Hiday 1992 Purdue Univerity Dissertation 
    <https://engineering.purdue.edu/people/kathleen.howell.1/Publications/dissertations/1992_Hiday.pdf>`_

    `Short 2010 Master's Thesis Purdue University 
    <https://engineering.purdue.edu/people/kathleen.howell.1/Publications/masters/2010_Short.pdf>`_

    Parameters
    ----------
    E: float
        eccentric anomaly
    mu: float
        mass ratio of system
    e: float
        eccen. of system

    Returns
    -------
    L_states: 5x6 array
        states of Lpoints ordered 1-5
    """
    
    A1, B1 = -1, +1
    A2, B2 = -1, -1
    A3, B3 = +1, +1

    xL1 = sci_op.bisect(f = lpx_func, a = 0, b = 1 - mu - 1e-5, args = (mu, A1, B1))
    xL2 = sci_op.bisect(f = lpx_func, a = 1 - mu + 1e-5, b = 3, args = (mu, A2, B2))
    xL3 = sci_op.bisect(f = lpx_func, a = -3, b = - mu - 1e-5, args = (mu, A3, B3))

    R = 1 - e*np.cos(E)

    Edot = 1/(1-e*np.cos(E))
    Rdot = -mu*e*np.sin(E)*Edot
    
    L1 = np.array([R*xL1, 0, 0, Rdot, 0, 0])
    L2 = np.array([R*xL2, 0, 0, Rdot , 0, 0])
    L3 = np.array([R*xL3, 0, 0, Rdot , 0, 0])
    L4 = np.array([1/2 - mu, + np.sqrt(3)/2, 0, 0 , 0, 0])
    L5 = np.array([1/2 - mu, - np.sqrt(3)/2, 0, 0 , 0, 0])
    
    L_states = np.vstack((L1, L2, L3, L4, L5))
    
    return L_states


def er3bp_JC(tau: float, state: list, mu: float, e: float):
    """
    er3bp_JC - calculate Jacobi constant in ER3BP model
    
    .. Note::
        The Jacobi constant is an integral of motion in the CR3BP
        The ER3BP does not possess a constant of integration from the 
        equations of motion. Nonetheless, it can be informative to compute 
        a particle's JC over a trajectory

    Source: Hiday 1992 Purdue Univerity Dissertation

    Parameters
    ----------
    tau: float
        non-dim time
    state: 1x6 array
        state vector
    mu: float
        mass ratio of system

    Returns
    -------
    JC: float
        Jacobi constant
    """

    x, y, z = state[0:3]
    v = np.linalg.norm(state[3:6])
    
    p1_state, p2_state = er3bp_pstates(tau, mu, e)
    r13 = np.linalg.norm(state[0:3] - p1_state[0:3])
    r23 = np.linalg.norm(state[0:3] - p2_state[0:3])

    U = 0.5*(x**2 + y**2) + (1-mu)/r13 + mu/r23
    
    JC = 2*U - v**2

    return JC

def er3bp_transform(tau: float, e: float):
    """
    er3bp_transform - computes transformation matrix from synodic to inertial frame

    General rotation matrix about z axis

    Parameters
    ----------
    tau: float
        epoch of transformation, this is equivalent to the true anomaly of the binary
        bounded [0, 2pi)
    e: float
        eccentricity of system

    Returns
    -------
    Q: array
        evaluated 6x6 rotation matrix Q(0, theta)
    """

    E = calc_ecc_anom(tau, e)
    # true anomaly is angle between barycenter and radial line
    f = calc_f_anom(E, e) 

    cos = np.cos(f)
    sin = np.sin(f)
    q = np.array([
        [cos, -sin, 0],
        [sin,  cos, 0],
        [  0,    0, 1]
    ])
    qdot = np.array([
        [-sin, -cos, 0],
        [cos,  -sin, 0],
        [  0,    0, 1]
    ])
    O3x3 = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])

    Qtop = np.concatenate((q, O3x3), axis = 1)
    Qbot = np.concatenate((qdot, q), axis = 1)

    Q = np.vstack((Qtop, Qbot))

    return Q
##########################################################
##################### BCR4BP Methods ######################
##########################################################
"""
Methods used for BCR4BP dynamical model

!!!!!!! This model is under development  !!!!!!

Methods
-------
bcr4bp_eom - calculates state derivative using bcr4bp eom
bcr4bp_jacobian - calculates jacobian matrix of er3bp EOM
bcr4bp_JC - calculate Jacobi constant in er3bp model
bcr4bp_pstates - calculates state of primaries in er3bp
bcr4bp_Lstates - calculates and returns array of Lagrange points in the er3bp
"""


def bcr4bp_eom(tau, s, mu, e, eval_STM = True, p_event = None, epoch = False):

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


    create_system('Mun', mu = 0.5)