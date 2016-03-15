#!/usr/bin/python

from sympy import Symbol, symbols, sin, cos, factor, simplify
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, time_derivative, divergence, gradient
from sympy.printing import ccode
from sympy.abc import t, rho
import string

# Declare coefficients and scalar variables as SymPy symbols
x = ReferenceFrame('x')                 # Spatial coordinates: x=x[0], y=x[1], z=x[2]
T0 = symbols('T0')                      # Minimum temperature
Ax, By, Cz = symbols('Ax By Cz')        # Spatial amplitudes for cosines
At, Bt, Ct, Dt = symbols('At Bt Ct Dt') # Temporal amplitudes for cosines
k0, k1, k2 = symbols('k0 k1 k2')        # Thermal conductivity quadratic coefficients
c0, c1, c2 = symbols('c0 c1 c2')        # Heat capacity quadratic coefficients


# Define the manufactured solution for Temperature as 1-, 2-, or 3-D scalar field.
# Modify as necessary, but make sure the domain boundaries satisfy periodicity.
T1 = T0 + cos(Ax*x[0] + At*t) * cos(Dt*t)
K1 = k0 + k1*T1 + k2*T1**2
C1 = c0 + c1*T1 + c2*T1**2

T2 = T0 + cos(Ax*x[0] + At*t) * cos(By*x[1] + Bt*t)*cos(Dt*t)
K2 = k0 + k1*T2 + k2*T2**2
C2 = c0 + c1*T2 + c2*T2**2

T3 = T0 + cos(Ax*x[0] + At*t) * cos(By*x[1] + Bt*t)*cos(Cz*x[2] + Ct*t)*cos(Dt*t)
K3 = k0 + k1*T3 + k2*T3**2
C3 = c0 + c1*T3 + c2*T3**2



# Do not change the following expressions! (unless you have a new heat law)
Q1 = rho*C1*time_derivative(T1,x) - divergence(K1*gradient(T1,x),x)
Q2 = rho*C2*time_derivative(T2,x) - divergence(K2*gradient(T2,x),x)
Q3 = rho*C3*time_derivative(T3,x) - divergence(K3*gradient(T3,x),x)


print "1D source:"
CQ1 = ccode(Q1)
CQ1 = CQ1.replace('x_x', 'x[0]')
print CQ1
print
print "2D source:"
CQ2 = ccode(Q2)
CQ2 = CQ2.replace('x_x', 'x[0]')
CQ2 = CQ2.replace('x_y', 'x[1]')
print CQ2
print
print "3D source:"
CQ3 = ccode(Q2)
CQ3 = CQ3.replace('x_x', 'x[0]')
CQ3 = CQ3.replace('x_y', 'x[1]')
CQ3 = CQ3.replace('x_z', 'x[2]')
print CQ3
