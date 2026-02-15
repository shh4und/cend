"""
Distance field computations for neuron tracing.

This module implements the 'pressure' and 'thrust' distance fields used in
DF-Tracing (Yang et al. 2013) for guiding skeleton generation.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.util import img_as_bool

logger = logging.getLogger(__name__)


class DistanceFields:
    """
    Manages distance field computations for neuron reconstruction.

    Implements pressure fields (distance from boundaries) and thrust fields
    (distance from seed point) used to guide skeleton tracing.
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        seed_point: Tuple[int, int, int] = (0, 0, 0),
        dataset_number: int = 0,
    ):
        """
        Initialize distance fields manager.

        Args:
            shape: Shape of the 3D volume (z, y, x).
            seed_point: Root/seed point coordinates.
            dataset_number: Dataset identifier for logging.
        """
        self.shape = shape
        self.seed_point = tuple(np.round(seed_point).astype(int).tolist())
        self.dataset_number = dataset_number
        self.logger = logger

    def pressure_field(self, mask: np.ndarray, metric: str = "euclidean") -> np.ndarray:
        """
        Computes the 'pressure' field (distance transform) for a given mask.

        The pressure field represents the distance from each foreground voxel
        to the nearest boundary, helping to identify medial axes.

        Args:
            mask: Binary mask of the neuron region.
            metric: Distance metric - "euclidean" or "taxicab".

        Returns:
            Distance transform of the mask.
        """
        neuron_mask = img_as_bool(mask)
        if metric == "euclidean":
            return ndi.distance_transform_edt(neuron_mask)
        elif metric == "taxicab":
            return ndi.distance_transform_cdt(neuron_mask, metric="taxicab")
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def thrust_field(
        self, mask: np.ndarray, seed_point: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Computes the 'thrust' field (Euclidean distance from seed) within a mask.

        The thrust field represents the distance from each voxel to the seed point,
        helping to identify terminal points and guide path finding.

        Args:
            mask: Binary mask of the neuron region.
            seed_point: Seed point coordinates (uses instance seed if None).

        Returns:
            Distance field from the seed point, masked to foreground only.
        """
        if seed_point is None:
            seed_point = self.seed_point

        seed_img = np.zeros(self.shape, dtype=bool)
        seed_img[seed_point] = True

        thrust_field = ndi.distance_transform_edt(~seed_img)
        thrust_field[~mask] = 0
        return thrust_field

    def find_thrust_maxima(
        self, thrust_field: np.ndarray, neuron_mask: np.ndarray, order: int = 1
    ) -> np.ndarray:
        """
        Finds local maxima in the thrust field.

        Local maxima in the thrust field correspond to terminal/end points
        of the neuron structure.

        Args:
            thrust_field: The thrust distance field.
            neuron_mask: Binary mask of the neuron region.
            order: Order of the local maximum filter (radius = order).

        Returns:
            Array of coordinates (N, 3) of local maxima points.
        """
        size = 1 + 2 * order
        footprint = np.ones((size, size, size))
        local_max = ndi.maximum_filter(thrust_field, footprint=footprint)

        # A voxel is a local maximum if it matches the maximum-filtered value
        # and is part of the foreground.
        maxima_mask = (thrust_field == local_max) & neuron_mask
        return np.argwhere(maxima_mask)

    def correct_and_update_root(
        self,
        skeleton_image: np.ndarray,
        original_root: Optional[Tuple[int, int, int]] = None,
    ) -> Optional[Tuple[int, int, int]]:
        """
        Validates if the root is on the skeleton. If not, finds the closest point.

        Args:
            skeleton_image: The binary skeleton image.
            original_root: The root point to check (uses instance seed if None).

        Returns:
            Tuple with the new valid root coordinates or None if skeleton is empty.
        """
        if not np.any(skeleton_image):
            self.logger.warning(f"OP_{self.dataset_number}: Empty skeleton image.")
            return None

        root = self.seed_point if original_root is None else original_root

        # Check if the original root is on a skeleton voxel.
        if skeleton_image[root]:
            self.logger.info(f"OP_{self.dataset_number}: Original root is inside skeleton.")
            return root

        skeleton_voxels = np.argwhere(skeleton_image)

        # Squared Euclidean distance (faster) to find the closest point.
        distances_sq = np.sum((skeleton_voxels - np.array(root)) ** 2, axis=1)

        # Find the index of the closest voxel.
        closest_voxel_index = np.argmin(distances_sq)
        new_valid_root = tuple(skeleton_voxels[closest_voxel_index].astype(int).tolist())

        self.seed_point = new_valid_root
        self.logger.info(f"OP_{self.dataset_number}: Root updated to: {new_valid_root}")
        return new_valid_root

    def get_26_neighborhood(self, voxel: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        Gets the 26-connected neighbors of a voxel within volume bounds.

        Args:
            voxel: The center voxel (z, y, x).

        Returns:
            List of valid neighbor coordinates.
        """
        z, y, x = voxel
        neighbors = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if (
                        0 <= nz < self.shape[0]
                        and 0 <= ny < self.shape[1]
                        and 0 <= nx < self.shape[2]
                    ):
                        neighbors.append((nz, ny, nx))
        return neighbors
