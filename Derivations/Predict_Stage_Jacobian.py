from sympy import sin, cos, tan, Matrix
from sympy.abc import theta, phi, p, q, r

x = Matrix([phi, theta])
f = Matrix([p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta),
            q * cos(phi) - r * sin(phi)])

# Jacobian Matrix
A = f.jacobian(x)
print(A)



