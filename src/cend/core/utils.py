"""
Morphological and image utility functions for neuron tracing.

This module collects small utility functions that are shared across the
pipeline but do not belong to a single algorithmic domain.
"""

import logging
from typing import Tuple

import numpy as np
from scipy import linalg
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte

logger = logging.getLogger(__name__)


def strel_non_flat_sphere(grey_morpho_weight: float = 0.5, grey_morpho_size: int = 2) -> np.ndarray:
    """
    Creates a non-flat spherical structuring element (paraboloid) for grey morphology.

    Spatial distances are first normalised to [0, 1] relative to the element size
    so that elements of different sizes remain comparable. The ``grey_morpho_weight``
    controls the depth of the resulting paraboloid independently of the size.

    Args:
        grey_morpho_weight: Depth of the paraboloid (controls erosion/dilation strength).
        grey_morpho_size: Half-size of the cube that bounds the structuring element.

    Returns:
        A 3-D float array representing the non-flat structuring element.
    """
    y, x, z = np.ogrid[
        -grey_morpho_size // 2 : grey_morpho_size // 2 + 1,
        -grey_morpho_size // 2 : grey_morpho_size // 2 + 1,
        -grey_morpho_size // 2 : grey_morpho_size // 2 + 1,
    ]
    dist_sq = (x**2 + y**2 + z**2).astype(float)
    max_dist_sq = dist_sq.max()
    if max_dist_sq > 0:
        dist_sq = dist_sq / max_dist_sq  # normalise to [0, 1]
    struct_nonflat = -dist_sq * grey_morpho_weight
    struct_nonflat[grey_morpho_size // 2, grey_morpho_size // 2, grey_morpho_size // 2] = 0
    return struct_nonflat


def create_maxima_image(coords: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Creates a binary uint8 image from a list of coordinates.

    Args:
        coords: An (N, 3) array of (z, y, x) coordinates.
        shape: The (z, y, x) shape of the output image.

    Returns:
        A binary uint8 image where pixels at the given coordinates are set to 255.
    """
    image = np.zeros(shape, dtype=np.uint8)
    image[tuple(coords.T)] = 255
    return img_as_ubyte(image)


def local_maxima_3d(data: np.ndarray, order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects local maxima in a 3D array.

    Args:
        data: The 3D array to search.
        order: Number of points on each side to consider for the comparison.

    Returns:
        Tuple of (coords, values) where coords is (N, 3) and values is (N,).
    """
    from scipy import ndimage as ndi

    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0  # exclude the centre pixel

    filtered = ndi.maximum_filter(data, footprint=footprint)
    mask = data > filtered

    coords = np.asarray(np.where(mask)).T
    values = data[mask]
    return coords, values


def compute_tubular_direction(
    volume: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Computes the principal tubular direction at every voxel from the Hessian.

    For each voxel, the eigenvector corresponding to the eigenvalue with the
    smallest absolute value indicates the direction along which curvature is
    minimal — i.e. the longitudinal axis of a tube.

    The computation is restricted to voxels where the Hessian trace is
    negative (bright tubular structures) to save memory and time.

    Args:
        volume: 3-D float array (e.g. pseudo-distance map or filtered response).
        sigma: Gaussian scale for computing the Hessian.

    Returns:
        A ``(Z, Y, X, 3)`` float32 array of unit direction vectors.
        Voxels outside the mask are set to zero.
    """
    scale = sigma**2
    vol = volume.astype(np.float64)

    Hxx = ndi.gaussian_filter(vol, sigma=sigma, order=[0, 0, 2]) * scale
    Hyy = ndi.gaussian_filter(vol, sigma=sigma, order=[0, 2, 0]) * scale
    Hzz = ndi.gaussian_filter(vol, sigma=sigma, order=[2, 0, 0]) * scale
    Hxy = ndi.gaussian_filter(vol, sigma=sigma, order=[0, 1, 1]) * scale
    Hxz = ndi.gaussian_filter(vol, sigma=sigma, order=[1, 0, 1]) * scale
    Hyz = ndi.gaussian_filter(vol, sigma=sigma, order=[1, 1, 0]) * scale

    trace = Hxx + Hyy + Hzz
    mask = (vol > 0) & (trace < 0)

    directions = np.zeros(vol.shape + (3,), dtype=np.float32)
    n_voxels = int(mask.sum())
    if n_voxels == 0:
        return directions

    H = np.empty((n_voxels, 3, 3), dtype=np.float64)
    H[:, 0, 0] = Hzz[mask]
    H[:, 1, 1] = Hyy[mask]
    H[:, 2, 2] = Hxx[mask]
    H[:, 0, 1] = H[:, 1, 0] = Hyz[mask]
    H[:, 0, 2] = H[:, 2, 0] = Hxz[mask]
    H[:, 1, 2] = H[:, 2, 1] = Hxy[mask]

    eigenvalues, eigenvectors = linalg.eigh(H)  # ascending order
    # smallest |eigenvalue| → index with min absolute value
    idx = np.argmin(np.abs(eigenvalues), axis=-1)
    # gather the corresponding eigenvector for each voxel
    tubular_dir = eigenvectors[np.arange(n_voxels), :, idx]
    # normalise (should already be unit but guard against edge cases)
    norms = np.linalg.norm(tubular_dir, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    tubular_dir /= norms

    directions[mask] = tubular_dir.astype(np.float32)
    return directions


def find_pseudo_distance_maxima(
    pseudo_distance: np.ndarray,
    neuron_mask: np.ndarray,
    order: int = 2,
    min_value: float = 0.0,
) -> np.ndarray:
    """
    Finds local maxima of a pseudo-distance (grey-erosion) map inside a mask.

    According to MAT theory, local maxima of the distance function correspond
    to medial-axis points.  Using these as seeds can reduce false positives
    that arise from irregular mask boundaries.

    Args:
        pseudo_distance: The grey-scale pseudo-distance field (float, 0–1).
        neuron_mask: Binary foreground mask.
        order: Half-window size for local maximum detection.
        min_value: Minimum pseudo-distance value to accept a maximum.

    Returns:
        ``(N, 3)`` integer array of seed coordinates.
    """
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    local_max = ndi.maximum_filter(pseudo_distance, footprint=footprint)
    maxima_mask = (pseudo_distance > local_max) & neuron_mask & (pseudo_distance > min_value)
    return np.argwhere(maxima_mask)
