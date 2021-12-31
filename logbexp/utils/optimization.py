
import numpy as np

from numpy.linalg import LinAlgError
from scipy.linalg import sqrtm
from scipy.optimize import minimize_scalar
from logbexp.utils.utils import sigmoid


def fit_online_logistic_estimate(arm, reward, current_estimate, vtilde_matrix, vtilde_inv_matrix, constraint_set_radius,
                                 diameter=1, precision=0.1):
    """
    ECOLog estimation procedure.
    """
    # some pre-computation
    sqrt_vtilde_matrix = sqrtm(vtilde_matrix)
    sqrt_vtilde_inv_matrix = sqrtm(vtilde_inv_matrix)
    z_theta_t = np.dot(sqrt_vtilde_matrix, current_estimate)
    z_estimate = z_theta_t
    inv_z_arm = np.dot(sqrt_vtilde_inv_matrix, arm)
    step_size = 1 / (1/4 + 1/(1 + diameter/2)) * 0.75 # slightly smaller step for stability
    iters = int((1/0.75) * np.ceil((9 / 4 + diameter / 8) * np.log(diameter / precision)))

    # few steps of projected gradient descent
    for _ in range(iters):
        pred_probas = sigmoid(np.sum(z_estimate * inv_z_arm))
        grad = z_estimate - z_theta_t + (pred_probas - reward) * inv_z_arm
        unprojected_update = z_estimate - step_size * grad
        z_estimate = project_ellipsoid(x_to_proj=unprojected_update,
                                       ell_center=np.zeros_like(arm),
                                       ecc_matrix=vtilde_matrix,
                                       radius=constraint_set_radius)
    theta_estimate = np.dot(sqrt_vtilde_inv_matrix, z_estimate)
    return theta_estimate


def fit_online_logistic_estimate_bar(arm, current_estimate, vtilde_matrix, vtilde_inv_matrix, constraint_set_radius,
                                     diameter=1, precision=0.1):
    """
    ECOLog estimation procedure to compute theta_bar.
    """
    # some pre-computation
    sqrt_vtilde_matrix = sqrtm(vtilde_matrix)
    sqrt_vtilde_inv_matrix = sqrtm(vtilde_inv_matrix)
    z_theta_t = np.dot(sqrt_vtilde_matrix, current_estimate)
    z_estimate = z_theta_t
    inv_z_arm = np.dot(sqrt_vtilde_inv_matrix, arm)
    step_size = 1 / (1 / 4 + 1 / (1 + diameter / 2)) * 0.75  # slightly smaller step for stability
    iters = int((1/0.75) * np.ceil((9 / 4 + diameter / 8) * np.log(diameter / precision)))

    #few steps of projected gradient descent
    for _ in range(iters):
        pred_probas = sigmoid(np.sum(z_estimate * inv_z_arm))
        grad = z_estimate - z_theta_t + (2*pred_probas - 1) * inv_z_arm
        unprojected_update = z_estimate - step_size * grad
        z_estimate = project_ellipsoid(x_to_proj=unprojected_update,
                                       ell_center=np.zeros_like(arm),
                                       ecc_matrix=vtilde_matrix,
                                       radius=constraint_set_radius)
    theta_estimate = np.dot(sqrt_vtilde_inv_matrix, z_estimate)
    return theta_estimate


def project_ellipsoid(x_to_proj, ell_center, ecc_matrix, radius, safety_check=False):
    """
    Orthogonal projection on ellipsoidal set
    :param x_to_proj: np.array(dim), point to project
    :param ell_center: np.array(dim), center of ellipsoid
    :param ecc_matrix: np.array(dimxdim), eccentricity matrix
    :param radius: float, ellipsoid radius
    :param safety_check: bool, check ecc_matrix psd
    """
    # start by checking if the point to project is already inside the ellipsoid
    ell_dist_to_center = np.dot(x_to_proj - ell_center, np.linalg.solve(ecc_matrix, x_to_proj - ell_center))
    is_inside = (ell_dist_to_center - radius ** 2) < 1e-3
    if is_inside:
        return x_to_proj

    # check eccentricity is symmetric PSD
    if safety_check:
        sym_check = np.allclose(ecc_matrix, ecc_matrix.T)
        psd_check = np.all(np.linalg.eigvals(ecc_matrix) > 0)
        if not sym_check or not psd_check:
            raise ValueError("Eccentricity matrix is not symetric or PSD")

    # some pre-computation
    dim = len(x_to_proj)
    sqrt_psd_matrix = sqrtm(ecc_matrix)
    y = np.dot(sqrt_psd_matrix, x_to_proj - ell_center)

    # opt function for projection
    def fun_proj(lbda):
        try:
            solve = np.linalg.solve(ecc_matrix + lbda * np.eye(dim), y)
            res = lbda * radius ** 2 + np.dot(y, solve)
        except LinAlgError:
            res = np.inf
        return res

    # find proj
    lbda_opt = minimize_scalar(fun_proj, method='bounded', bounds=(0, 1000), options={'maxiter': 500})
    eta_opt = np.linalg.solve(ecc_matrix + lbda_opt.x * np.eye(dim), y)
    x_projected = np.dot(sqrt_psd_matrix, eta_opt) + ell_center

    return x_projected
