from sympy import Matrix
from sympy.abc import a, b, c, d, l, m, n, o, w, x, y, z, t, v

# Covariance matrix
P = Matrix([a, b, c, d]).reshape(rows=2, cols=2)
print(P)
C = Matrix([l, m, n, o, t, v]).reshape(rows=3, cols=2)
print(C)
R = Matrix([w, x, y, z]).reshape(rows=2, cols=2)
print(R)

C_T = C.T
print(C_T)

# [C*P*C'+R]
G = C * P * C.T# + R
print("G = {}nSize: {}".format(G, G.shape))

# G_inv = G.inv()
# print("G_inv = {}".format(G_inv))
#
# # [K = P*C^T * [C*P*C^T+R]^-1]
# K = P * C.T * G_inv
# print(K)
