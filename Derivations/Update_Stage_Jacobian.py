from sympy import sin, cos, Matrix
from sympy.abc import theta, phi, g

x = Matrix([phi, theta])
h = Matrix([g * sin(theta), g * (-cos(theta) * sin(phi)), g * (-cos(theta) * cos(phi))])

J = h.jacobian(x)
print(J)
print(J.shape)
