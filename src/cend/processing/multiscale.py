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


def _compute_multiscale(
    volume: np.ndarray,
    sigma_range: Tuple[float, float, float],
    filter_type: str,
    neuron_threshold: float,
    dataset_number: int,
    label: str,
) -> np.ndarray:
    """
    Core multi-scale filtering loop shared by all public wrappers.

    Builds a geometric progression of scales from ``sigma_range`` and takes
    the per-voxel maximum response across all scales.

    Args:
        volume: The input 3D volume (already float).
        sigma_range: Tuple of (min, max, ratio). ``ratio`` must be > 1 for
            multiple scales; otherwise only ``sigma_min`` is used.
        filter_type: Type of filter — "yang", "frangi", or "kumar".
        neuron_threshold: Threshold parameter forwarded to the filter.
        dataset_number: Dataset identifier used in log messages.
        label: Short descriptive label used in log messages (e.g. "Filtering").

    Returns:
        Max-response volume as float in the same shape as ``volume``.
    """
    sigma_min, sigma_max, sigma_step = sigma_range
    if sigma_min <= 0 or sigma_step <= 1 or sigma_min >= sigma_max:
        sigmas = np.array([sigma_min], dtype=float)
    else:
        K = int(np.floor(np.log(sigma_max / sigma_min) / np.log(sigma_step))) + 1
        sigmas = sigma_min * sigma_step ** np.arange(K)
        sigmas = sigmas[sigmas <= sigma_max + 1e-9]
        sigmas = np.round(sigmas, 2)

    logger.info(f"OP_{dataset_number}: Starting Multiscale {label} with {filter_type}'s approach.")
    logger.info(f" params: sigma_scale: {sigmas}, neuron_thresh: {neuron_threshold}")

    max_response = np.zeros_like(volume, dtype=float)
    for sig in sigmas:
        logger.info(f"OP_{dataset_number} -> Filtering at scale: {sig}")
        max_response = np.maximum(
            max_response, apply_tubular_filter(volume, sig, filter_type, neuron_threshold)
        )

    logger.info(f"OP_{dataset_number}: Multiscale {label} complete.")
    return img_as_float(max_response)


def multiscale_filtering(
    volume: np.ndarray,
    sigma_range: Tuple[float, float, float] = (1, 4, 1.5),
    filter_type: str = "yang",
    neuron_threshold: float = 0.05,
    dataset_number: int = 0,
) -> np.ndarray:
    """
    Applies the anisotropic filter across a range of scales on the raw volume.

    The final response for each voxel is the maximum response found across
    all scales, allowing detection of neurites with varying thicknesses.

    Args:
        volume: The input 3D volume.
        sigma_range: Tuple of (min, max, ratio) defining a geometric progression
            of scales.
        filter_type: Type of filter — "yang", "frangi", or "kumar".
        neuron_threshold: Threshold parameter for filtering.
        dataset_number: Dataset identifier for logging.

    Returns:
        The maximum response volume from multi-scale filtering.
    """
    return _compute_multiscale(
        img_as_float(volume),
        sigma_range,
        filter_type,
        neuron_threshold,
        dataset_number,
        label="Filtering",
    )


def multiscale_on_distance(
    volume: np.ndarray,
    sigma_range: Tuple[float, float, float] = (1, 2, 1.15),
    filter_type: str = "frangi",
    neuron_threshold: float = 0.05,
    dataset_number: int = 0,
) -> np.ndarray:
    """
    Applies the anisotropic filter across a range of scales on a distance-transformed volume.

    Identical in structure to ``multiscale_filtering`` but intended for a second
    filtering pass on a morphologically-processed input, with different defaults
    suited for the distance-field stage of the pipeline.

    Args:
        volume: The input 3D volume (typically after grey closing).
        sigma_range: Tuple of (min, max, ratio) defining a geometric progression
            of scales.
        filter_type: Type of filter — "yang", "frangi", or "kumar".
        neuron_threshold: Threshold parameter for filtering.
        dataset_number: Dataset identifier for logging.

    Returns:
        The maximum response volume from multi-scale filtering.
    """
    return _compute_multiscale(
        img_as_float(volume),
        sigma_range,
        filter_type,
        neuron_threshold,
        dataset_number,
        label="On Distance",
    )
