"""Local Fisher information in natural (KM, Vmax) coordinates."""

import numpy as np
from numpy.typing import ArrayLike

from enzymeopt.model import michaelis_menten_jacobian


def fisher_information(concentrations: ArrayLike, *, km: float, vmax: float,
                       noise_std: float = 1.0) -> np.ndarray:
    """Return sum(g g.T) / sigma**2; empty designs have zero information.

    Sigma must be positive. For noiseless design ranking use unit variance:
    a common positive variance changes scores, but not their ordering.
    """
    if isinstance(noise_std, (bool, np.bool_)) or not np.isscalar(noise_std):
        raise ValueError("noise_std must be positive and finite")
    if not np.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("noise_std must be positive and finite")
    raw = np.asarray(concentrations)
    if raw.ndim != 1 or raw.dtype.kind not in "fiu":
        raise ValueError("concentrations must be a one-dimensional real numeric array")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            jac = michaelis_menten_jacobian(raw, km, vmax) / noise_std
            matrix = jac.T @ jac
        except FloatingPointError as error:
            raise ValueError("information cannot be represented at this parameter scale") from error
    if not np.all(np.isfinite(matrix)):
        raise ValueError("information must be finite")
    return matrix


def information_logdet(matrix: ArrayLike) -> float:
    """Stable log determinant of a PSD 2x2 matrix; singular returns -inf.

    Numerically unresolved eigenvalues (relative to machine precision) count
    as singular. No ridge or prior information is silently added.
    """
    raw = np.asarray(matrix)
    if raw.shape != (2, 2) or raw.dtype.kind not in "fiu":
        raise ValueError("information must be a real 2 by 2 matrix")
    values = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("information must be finite")
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return float("-inf")
    normalized = values / scale
    tolerance = 16 * np.finfo(float).eps
    if not np.allclose(normalized, normalized.T, atol=tolerance, rtol=0):
        raise ValueError("information must be symmetric")
    eigenvalues = np.linalg.eigvalsh((normalized + normalized.T) / 2)
    if eigenvalues[0] < -tolerance:
        raise ValueError("information must be positive semidefinite")
    if eigenvalues[0] <= tolerance:
        return float("-inf")
    return float(np.log(eigenvalues).sum() + 2 * np.log(scale))
