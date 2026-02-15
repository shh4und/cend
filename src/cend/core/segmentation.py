"""
Image segmentation and denoising methods for 3D neuron volumes.

This module provides thresholding and morphological operations for segmenting
and cleaning neuron masks.
"""

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.util import img_as_bool, img_as_float


def adaptive_mean_mask(
    volume: np.ndarray,
    zero_t: bool = False,
    tol: float = 1e-3,
    max_iterations: int = 100,
) -> Tuple[np.ndarray, float]:
    """
    Generates a binary mask using iterative adaptive mean thresholding.

    This method determines a global threshold by iteratively averaging the
    mean intensities of pixels above and below the current threshold.

    Args:
        volume: The input volume to be thresholded.
        zero_t: If True, simply thresholds at 0.
        tol: Convergence tolerance for the threshold.
        max_iterations: Maximum number of iterations.

    Returns:
        Tuple of (binary boolean mask, final threshold value).
    """
    if zero_t:
        return volume > 0, 0.0

    current_threshold = np.mean(volume)

    for _ in range(max_iterations):
        higher_mask = volume > current_threshold
        lower_mask = ~higher_mask

        if not np.any(higher_mask) or not np.any(lower_mask):
            break

        mean_higher = np.sum(volume[higher_mask]) / np.count_nonzero(higher_mask)
        mean_lower = np.sum(volume[lower_mask]) / np.count_nonzero(lower_mask)

        new_threshold = (mean_lower + mean_higher) / 2

        if abs(new_threshold - current_threshold) < tol:
            break

        current_threshold = new_threshold

    final_mask = volume > current_threshold
    return final_mask, current_threshold


def apply_mask(mask: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Applies a binary mask to the volume, setting non-masked voxels to 0.

    Args:
        mask: Binary mask to apply.
        volume: Volume to be masked.

    Returns:
        Masked volume as float.
    """
    mask = img_as_bool(mask)
    target_volume = volume.copy()

    if mask.shape != target_volume.shape:
        raise ValueError("Mask and volume must have the same shape.")

    target_volume[~mask] = 0
    return img_as_float(target_volume)


def morphological_denoising(
    neuron_mask: np.ndarray, structure: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Removes salt-and-pepper noise using morphological opening and closing.

    Args:
        neuron_mask: Binary mask to denoise.
        structure: Structuring element (default: 3D cross).

    Returns:
        Denoised binary mask.
    """
    strel = ndi.generate_binary_structure(3, 1) if structure is None else structure

    mask = img_as_bool(
        ndi.binary_closing(ndi.binary_opening(neuron_mask, structure=strel), structure=strel)
    )
    return mask


def grey_morphological_denoising(
    neuron_mask: np.ndarray, structure: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Removes salt-and-pepper noise using grey morphological opening and closing.

    Args:
        neuron_mask: Grayscale image to denoise.
        structure: Structuring element (default: 3x3x3 box).

    Returns:
        Denoised grayscale image.
    """
    mask = ndi.grey_closing(ndi.grey_opening(neuron_mask, size=(3, 3, 3)), size=(3, 3, 3))
    return mask


def gradient_magnitude(volume: np.ndarray) -> np.ndarray:
    """
    Computes the gradient magnitude of the volume using Sobel operators.

    Args:
        volume: Input 3D volume.

    Returns:
        Normalized gradient magnitude.
    """
    sobel_z = ndi.sobel(volume, 0)
    sobel_y = ndi.sobel(volume, 1)
    sobel_x = ndi.sobel(volume, 2)
    magnitude = np.sqrt(np.square(sobel_z) + np.square(sobel_y) + np.square(sobel_x))
    grad_magnitude_max = magnitude.max()
    if grad_magnitude_max != 0:
        magnitude /= grad_magnitude_max
    return magnitude


def boundary_voxels(volume: np.ndarray) -> np.ndarray:
    """
    Finds the boundary voxels of a binary mask using morphological erosion.

    Args:
        volume: Binary volume.

    Returns:
        Binary image with only boundary voxels.
    """
    mask = volume.astype(bool)
    struct_element = ndi.generate_binary_structure(rank=3, connectivity=3)
    eroded_mask = ndi.binary_erosion(mask, structure=struct_element)
    return mask ^ eroded_mask
