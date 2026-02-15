"""
Multi-scale filtering orchestration for neuron enhancement.

This module provides high-level functions for applying tubular filters
across multiple scales and combining results.
"""

import logging
from typing import Tuple

import numpy as np
from skimage.util import img_as_float

from ..core.filters import apply_tubular_filter

logger = logging.getLogger(__name__)


def multiscale_filtering(
    volume: np.ndarray,
    sigma_range: Tuple[float, float, float] = (1, 4, 1),
    filter_type: str = "yang",
    neuron_threshold: float = 0.05,
    dataset_number: int = 0,
) -> np.ndarray:
    """
    Applies the anisotropic filter across a range of scales (sigmas).

    The final response for each voxel is the maximum response found across
    all scales, allowing detection of neurites with varying thicknesses.

    Args:
        volume: The input 3D volume.
        sigma_range: Tuple of (min, max, step) for the sigma values.
        filter_type: Type of filter - "yang", "frangi", "kumar", or "sato".
        neuron_threshold: Threshold parameter for filtering.
        dataset_number: Dataset identifier for logging.

    Returns:
        The maximum response volume from multi-scale filtering.
    """
    volume = img_as_float(volume)

    sigma_min, sigma_max, sigma_step = sigma_range
    sigmas = np.arange(sigma_min, sigma_max + sigma_step, sigma_step)
    if not sigmas.size:
        sigmas = np.array([sigma_min], dtype=float)

    max_response_volume = np.zeros_like(volume, dtype=float)

    logger.info(
        f"OP_{dataset_number}: Starting Multiscale Filtering with {filter_type}'s approach."
    )
    logger.info(f" params: sigma_scale: {sigmas}, neuron_thresh: {neuron_threshold}")

    for sig in sigmas:
        logger.info(f"OP_{dataset_number} -> Filtering at scale: {sig}")
        current_response = apply_tubular_filter(volume, sig, filter_type, neuron_threshold)
        max_response_volume = np.maximum(max_response_volume, current_response)

    logger.info(f"OP_{dataset_number}: Multiscale Filtering complete.")
    return img_as_float(max_response_volume)
