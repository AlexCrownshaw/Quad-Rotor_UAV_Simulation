from sympy import Matrix
from sympy.abc import T, a, b, c, d, l, m, n, o, w, x, y, z

# Covariance matrix
P = Matrix([a, b, c, d]).reshape(rows=2, cols=2)
print(P)
Q = Matrix([l, m, n, o]).reshape(rows=2, cols=2)
print(Q)
A = Matrix([w, x, y, z]).reshape(rows=2, cols=2)

# P_n+1 = P_n + T * (A*P_n + P_n*A^T + Q)
print(P + T * (A * P + P * A.T + Q))
