.. pyraa documentation master file, created by
   sphinx-quickstart on Thu Jun 23 18:05:37 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br />

.. .. image:: PyDAA_logo.png
..    :width: 510px
..    :height: 300px
..    :align: center

PyRAA |br| Python Restricted Astronomy and Astrodynamics
========================================================

**Open-source Python package for dynamical systems analysis of restricted multi-body models**

Thanks for checking out ``PyRAA``! The Python Restricted Astronomy and Astrodynamics library was developed
to aid researchers in dynamical astronomy and multi-body astrodynamics with routine calculations in restricted 
multi-body models. 

**PyRAA Core Functionality**
   * Circular and Elliptical Restricted Three-body dynamical models 
   * Variable precision Runge-Kutta numerical integration 
   * Flexible and advanced plotting capabilities
   * Differential corrections targeting algorithms
   * Numerical continuation schemes
   * Poincaré map generation

**What is a restricted multi-body model?**

A restricted multi-body model is a simplification of the gravitational N-body problem. The model assumes that 
the particle of interest is massless and subject to the gravitational influce of two or more other bodies. 
The massive bodies move in predictable 2-body motion, while the particle of interest is studied under the 
dynamics of two or more gravitational interactions. 

Examples:

* Circular Restricted Three-body Problem (CRT3BP)
* Elliptical Restricted Three-body Problem (ERT3BP)
* Bicircular Restricted Four-body Problem (BCR4BP)

**What are restricted multi-body models used for?**

Restricted multi-body models are most useful when studying the effects of multiple gravitational fields 
on a body that is relatively small compared to the influencing masses.

* Earth-Moon spacecraft trajectory design 
* Circumbinary exoplanet orbital dynamics 
* Dust particles in at solar system with a giant planet

.. important:: 
   New Features!
      * ``numba`` accelerated computations, integration time peformance boost, *10x faster*
      * ``pyraa.Targeter`` object oriented differential corrections and eigenspectrum analysis
      * inertial coordinates saved as ``sat`` attribute

.. note::
   The primary use-case for ``PyRAA`` is dynamical systems analysis of restricted multi-body models used in astrodynamics 
   and dynamical astronomy. 

   ``PyRAA`` does not have functionality similar to other orbital dynamics software such as ``REBOUND``,
   ``poliastro``, ``STK``, etc.

   **Not supported features**

   * N-body integration
   * 2-body orbital dynamics
   * Lambert solver
   * General coordinate transformations

:Author: 
   Drew Langford

:Email: 
   langfora@purdue.edu


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials.rst
   api.rst
   acknowldegments.rst
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
