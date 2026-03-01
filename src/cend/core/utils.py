"""
Morphological and image utility functions for neuron tracing.

This module collects small utility functions that are shared across the
pipeline but do not belong to a single algorithmic domain.
"""

from typing import Tuple

import numpy as np
from skimage.util import img_as_ubyte


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
