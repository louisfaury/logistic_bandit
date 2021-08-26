
import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import sqrtm
from scipy.optimize import brute

"""
Small script to evaluate procedure for ellipsoidal projection
"""

# initialization
psd_matrix = np.array([[2, -1], [-1, 2]])
norm_constraint = 2
x_to_proj = np.array([-7, 2])

# some pre-computation
sqrt_psd_matrix = sqrtm(psd_matrix)
y = np.dot(sqrt_psd_matrix, x_to_proj)


# opt function for projection
def fun_proj(lbda):
    return lbda*norm_constraint**2 + np.dot(y, np.linalg.solve(psd_matrix+lbda*np.eye(2), y))

# find proj
lbda_opt = brute(fun_proj, [(1e-12, 10)], Ns=4000)
print(lbda_opt)
eta_opt = np.linalg.solve(psd_matrix+lbda_opt*np.eye(2), y)
x_projected = np.dot(sqrt_psd_matrix, eta_opt)
print(x_to_proj)
print(x_projected)
print(np.linalg.norm(np.linalg.solve(sqrt_psd_matrix, x_projected)))

# plot to check
fig, ax = plt.subplots(1,1)
ax.axis('equal')
# start with the ellipsoid
u = np.array([[np.cos(angle), np.sin(angle)] for angle in np.linspace(0, 2*np.pi, 100)]).T
ell_ctr = norm_constraint*np.dot(sqrt_psd_matrix, u)
plt.plot(ell_ctr[0, :], ell_ctr[1, :])
# now plot projections
plt.plot(x_to_proj[0], x_to_proj[1], 'ro')
plt.plot(x_projected[0], x_projected[1], 'bx')
plt.plot([x_to_proj[0], x_projected[0]], [x_to_proj[1], x_projected[1]], '-')
plt.show()

