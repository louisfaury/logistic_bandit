
import matplotlib.pyplot as plt
import numpy as np

from numpy.linalg import LinAlgError
from scipy.linalg import sqrtm
from scipy.optimize import minimize_scalar
from utils.optimization import project_ellipsoid

"""
Small script to evaluate procedure for ellipsoidal projection
"""

# test hps
psd_matrix = np.array([[2, -1], [-1, 2]])  # must be PSD !
norm_constraint = 2
x_to_proj = np.array([-7, 5])
center = np.array([-2, 3])

# do proj
x_projected = project_ellipsoid(x_to_proj, center, psd_matrix, norm_constraint, safety_check=True)

# numerical check
proj_res = np.dot(x_projected-center, np.linalg.solve(psd_matrix, x_projected-center))-norm_constraint**2
print("{} (expect True)".format(np.isclose(proj_res, 0, atol=1e-03)))

# plot to check
fig, ax = plt.subplots(1, 1)
ax.axis('equal')
# start with the ellipsoid
u = np.array([[np.cos(angle), np.sin(angle)] for angle in np.linspace(0, 2*np.pi, 300)]).T
ell_ctr = center[:, None] + norm_constraint*np.dot(sqrtm(psd_matrix), u)
plt.plot(ell_ctr[0, :], ell_ctr[1, :])
# now plot projections
plt.plot(x_to_proj[0], x_to_proj[1], 'ro')
plt.plot(x_projected[0], x_projected[1], 'bx')
plt.plot([x_to_proj[0], x_projected[0]], [x_to_proj[1], x_projected[1]], '-')
plt.show()

