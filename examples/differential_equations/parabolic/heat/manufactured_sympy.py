#!/usr/bin/python

from sympy import Symbol, symbols, sin, cos, factor, simplify
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, time_derivative, divergence, gradient
from sympy.printing import ccode
from sympy.abc import t, rho
import string

# Declare space and time
x = ReferenceFrame('x') # Replaces non-physics expression, x, y, z, t = symbols("x y z t")
#t = Symbol('t')

# Declare coefficients and scalar variables
Ax, By, Cz = symbols('Ax By Cz')
At, Bt, Ct, Dt = symbols('At Bt Ct Dt')
T0 = symbols('T0')
k0, k1, k2 = symbols('k0 k1 k2')
c0, c1, c2 = symbols('c0 c1 c2')

# Define Temperature as 1-, 2-, or 3-D scalar field
#    Basis vectors are x.x, x.y, x.z (i.e., \hat{i} etc.)
#    Coordinate variables are x[0], x[1], x[2]

T1 = T0 * (1 + cos(Ax*x[0] + At*t) * cos(Dt*t))
K1 = k0 + k1*T1 + k2*T1**2
C1 = c0 + c1*T1 + c2*T1**2
Q1 = rho*C1*time_derivative(T1,x) - divergence(K1*gradient(T1,x),x)

T2 = T0 * (1 + cos(Ax*x[0] + At*t) * cos(By*x[1] + Bt*t)*cos(Dt*t))
K2 = k0 + k1*T2 + k2*T2**2
C2 = c0 + c1*T2 + c2*T2**2
Q2 = rho*C2*time_derivative(T2,x) - divergence(K2*gradient(T2,x),x)


T3 = T0 * (1 + cos(Ax*x[0] + At*t) * cos(By*x[1] + Bt*t)*cos(Cz*x[2] + Ct*t)*cos(Dt*t))
K3 = k0 + k1*T3 + k2*T3**2
C3 = c0 + c1*T3 + c2*T3**2
Q3 = rho*C3*time_derivative(T3,x) - divergence(K3*gradient(T3,x),x)


# Declare expression for Q1 from Mathematica for validation purposes
MQ1 = T0*(Ax**2*k0*cos(Dt*t)*cos(At*t + Ax*x[0]) + Ax**2*k1*T0*cos(Dt*t)*cos(At*t + Ax*x[0]) + 
      Ax**2*k2*T0**2*cos(Dt*t)*cos(At*t + Ax*x[0]) + Ax**2*k1*T0*cos(Dt*t)**2*cos(At*t + Ax*x[0])**2 + 
      2*Ax**2*k2*T0**2*cos(Dt*t)**2*cos(At*t + Ax*x[0])**2 + 
      Ax**2*k2*T0**2*cos(Dt*t)**3*cos(At*t + Ax*x[0])**3 - c0*Dt*rho*cos(At*t + Ax*x[0])*sin(Dt*t) - 
      c1*Dt*rho*T0*cos(At*t + Ax*x[0])*sin(Dt*t) - c2*Dt*rho*T0**2*cos(At*t + Ax*x[0])*sin(Dt*t) - 
      c1*Dt*rho*T0*cos(Dt*t)*cos(At*t + Ax*x[0])**2*sin(Dt*t) - 
      2*c2*Dt*rho*T0**2*cos(Dt*t)*cos(At*t + Ax*x[0])**2*sin(Dt*t) - 
      c2*Dt*rho*T0**2*cos(Dt*t)**2*cos(At*t + Ax*x[0])**3*sin(Dt*t) - At*c0*rho*cos(Dt*t)*sin(At*t + Ax*x[0]) - 
      At*c1*rho*T0*cos(Dt*t)*sin(At*t + Ax*x[0]) - At*c2*rho*T0**2*cos(Dt*t)*sin(At*t + Ax*x[0]) - 
      At*c1*rho*T0*cos(Dt*t)**2*cos(At*t + Ax*x[0])*sin(At*t + Ax*x[0]) - 
      2*At*c2*rho*T0**2*cos(Dt*t)**2*cos(At*t + Ax*x[0])*sin(At*t + Ax*x[0]) - 
      At*c2*rho*T0**2*cos(Dt*t)**3*cos(At*t + Ax*x[0])**2*sin(At*t + Ax*x[0]) - 
      Ax**2*k1*T0*cos(Dt*t)**2*sin(At*t + Ax*x[0])**2 - 
      2*Ax**2*k2*T0**2*cos(Dt*t)**2*sin(At*t + Ax*x[0])**2 - 
      2*Ax**2*k2*T0**2*cos(Dt*t)**3*cos(At*t + Ax*x[0])*sin(At*t + Ax*x[0])**2)

print
print "Is Q1 equivalent in SymPy and Mathematica?"
if simplify(MQ1 - Q1) ==0:
	print "Yes :)"
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
else:
	print "No :("
