import numpy as np
from scipy import ndimage as ndi
from scipy import linalg
import logging
from skimage.util import img_as_float, img_as_bool
from typing import Tuple, Optional, Dict, List, Set
import heapq


class DistanceFields:
    """
    Implements the DF-Tracing method for automatic 3D neuron reconstruction.

    This class provides tools for the first two steps of the DF-Tracing
    algorithm described in "A distance-field based automatic neuron tracing
    method" by Yang et al. (2013). This includes:
    1.  Image preprocessing with multi-scale anisotropic filtering.
    2.  Generation of distance fields for skeletonization.

    Attributes:
        volume (np.ndarray): The input 3D image volume, converted to float.
        sigmas (np.ndarray): An array of sigma values for multi-scale analysis.
        step_size (float): The step size used in path-finding algorithms.
        neuron_threshold (float): The threshold to segment neuron from background.
    """

    def __init__(
        self,
        volume: np.ndarray,
        filter_type: str = "yang",
        sigma_range: Tuple[float, float, float] = (1, 4, 1),
        step_size: float = 1.0,
        neuron_threshold: float = 0.05,
        seed_point: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        dataset_number: int = 0,
    ):
        """
        Initializes the DistanceFields class.

        Args:
            volume (np.ndarray): The 3D input image data.
            sigma_range (tuple): A tuple (min, max, step) for the sigma values.
            step_size (float): The step size for tracing algorithms.
            neuron_threshold (float): Initial threshold for neuron segmentation.
        """
        self.volume = img_as_float(volume.copy())
        self.shape = volume.shape

        sigma_min, sigma_max, sigma_step = sigma_range
        self.sigmas = np.arange(sigma_min, sigma_max + sigma_step, sigma_step)
        if not self.sigmas.size:
            self.sigmas = np.array([sigma_min], dtype=float)

        self.step_size = step_size
        self.neuron_threshold = neuron_threshold
        self.seed_point = tuple(np.round(seed_point).tolist())
        self.skeleton: np.ndarray = np.zeros_like(volume.shape)
        self.filter_type = filter_type
        self.dataset_number = dataset_number
        self._volume_max = self.volume.max()
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_skeleton(self, skel: np.ndarray):
        """Sets the skeleton image."""
        self.skeleton = skel

    def get_skeleton(self):
        """Returns the current skeleton image."""
        return self.skeleton

    def _yang_tubularity(
        self, hessian: np.ndarray, point: Tuple[int, int, int], sigma: float
    ) -> float:
        """
        Calculates the tubularity measure based on Hessian eigenvalues.

        This implements the vesselness function f(u) from Yang et al. (2013) to
        enhance tube-like structures. The function is high when one eigenvalue
        is close to zero and the other two are large and negative.

        Args:
            hessian (np.ndarray): The 3x3 Hessian matrix.
            point (tuple): The coordinates of the current voxel.
            sigma (float): The scale at which the Hessian was computed.

        Returns:
            float: The tubularity score.
        """
        eigenvalues = self._compute_eigenvalues(hessian, point, sigma)[0]
        if eigenvalues is None:
            return 0.0

        lambda1, lambda2, lambda3 = eigenvalues

        # Condition for bright, line-like structures:
        # lambda1 is near 0, while lambda2 and lambda3 are negative.
        if lambda2 >= 0 or lambda3 >= 0 or np.abs(lambda1) > self.neuron_threshold:
            return 0.0

        alphas = np.array([0.5, 0.5, 25.0])  # Coefficients from the reference paper
        sum_lambda_sq = np.sum(np.square(eigenvalues))
        if sum_lambda_sq == 0:
            return 0.0

        # f(u) from Equation 2 in the paper.
        # The paper may have a typo; a factor of 2 is common here.
        k_factors = np.exp(-(np.square(eigenvalues) / (2 * sum_lambda_sq)))
        tubularity = np.sum(alphas * k_factors)

        return tubularity

    def _frangi_vesselness(
        self,
        hessian: np.ndarray,
        point: Tuple[int, int, int],
        sigma: float,
        alpha: float = 0.5,
        beta: float = 0.5,
        c: float = 500.0,
    ) -> float:
        """
        Calculates the vesselness response using the Frangi et al. (1998) filter.
        """
        # 1. Get eigenvalues sorted by absolute value
        eigenvalues, _ = self._compute_eigenvalues(
            hessian, point, sigma, sort_by="abs"
        )
        if eigenvalues is None:
            return 0.0

        # For bright tubular structures on a dark background, the two largest
        # absolute eigenvalues (corresponding to eigenvalues[1] and eigenvalues[2])
        # must be negative.
        if eigenvalues[1] >= 0 or eigenvalues[2] >= 0:
            return 0.0

        lambda1_abs, lambda2_abs, lambda3_abs = np.abs(eigenvalues)
        epsilon = 1e-10

        # 2. Define Frangi filter terms
        # Ratio to distinguish between plate-like and line-like structures
        Ra = lambda2_abs / (lambda3_abs + epsilon)
        # Ratio to distinguish between blob-like and line-like structures
        Rb = lambda1_abs / (np.sqrt(lambda2_abs * lambda3_abs) + epsilon)
        # Norm of the Hessian matrix to account for overall structure intensity
        S = np.sqrt(np.sum(np.square(eigenvalues)))

        # 3. Calculate the filter response
        exp_Ra = 1.0 - np.exp(-(Ra**2) / (2 * alpha**2))
        exp_Rb = np.exp(-(Rb**2) / (2 * beta**2))
        exp_S = 1.0 - np.exp(-(S**2) / (2 * c**2))

        # The vesselness is the product of these three terms
        vesselness = exp_Ra * exp_Rb * exp_S

        return vesselness

    def _kumar_vesselness(
        self,
        hessian: np.ndarray,
        point: Tuple[int, int, int],
        sigma: float,
        c_factor: float = 0.5,
    ) -> float:
        """
        Calculates the vesselness response using the Kumar et al. (2013) filter.

        This is based on the Multi-scale Vessel Enhancement Filter (MVEF)
        from the reference paper.
        """
        eigenvalues, _ = self._compute_eigenvalues(
            hessian, point, sigma, sort_by="abs"
        )
        if eigenvalues is None:
            return 0.0


        lambda1, lambda2, lambda3 = eigenvalues
        abs_l2, abs_l3 = np.abs(lambda2), np.abs(lambda3)
        epsilon = 1e-10


        term1 = 1.0 - (np.abs(abs_l2 - abs_l3) / (abs_l2 + abs_l3 + epsilon))
        term2 = (2.0 / 3.0) * lambda1 - lambda2 - lambda3
        S_structureness = np.sqrt(np.sum(np.square(eigenvalues)))

        c_const = c_factor * self._volume_max
        if c_const < epsilon:
            c_const = 0.5

        term3_structureness = 1.0 - np.exp(
            -(S_structureness) / (2 * c_const**2 + epsilon)
        )

        v_sigma = term1 * term2 * term3_structureness

        if v_sigma <= 0:
            return 0.0
        else:
            k = 0.5
            vesselness = v_sigma * np.exp(k * sigma)
            return vesselness

    def tubular_filter(
        self, volume: Optional[np.ndarray] = None, sigma: Optional[float] = None
    ) -> np.ndarray:
        """
        Applies a Hessian-based anisotropic filter to enhance tubular structures.

        This implements v = exp(-||∇u||²) * f(u) from the paper, where f(u) is
        the tubularity score from `_yang_tubularity`.

        Args:
            volume (np.ndarray, optional): The input volume. Defaults to class volume.
            sigma (float, optional): The scale for Gaussian derivatives.
                                     Defaults to the minimum sigma of the class.

        Returns:
            np.ndarray: The filtered image.
        """
        if volume is None:
            volume = self.volume.copy()
        if sigma is None:
            sigma = self.sigmas[0]

        filtered_image = np.zeros_like(volume, dtype=float)

        # Compute second-order derivatives (Hessian components)
        h_xx = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 0, 2])
        h_yy = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 2, 0])
        h_zz = ndi.gaussian_filter(volume, sigma=sigma, order=[2, 0, 0])
        h_xy = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 1, 1])
        h_xz = ndi.gaussian_filter(volume, sigma=sigma, order=[1, 0, 1])
        h_yz = ndi.gaussian_filter(volume, sigma=sigma, order=[1, 1, 0])

        non_zero_voxels = np.nonzero(volume)
        for z, y, x in zip(*non_zero_voxels):
            point = (z, y, x)

            trace = h_zz[z, y, x] + h_yy[z, y, x] + h_xx[z, y, x]
            if trace >= 0:
                continue

            hessian_matrix = np.array(
                [
                    [h_zz[z, y, x], h_yz[z, y, x], h_xz[z, y, x]],
                    [h_yz[z, y, x], h_yy[z, y, x], h_xy[z, y, x]],
                    [h_xz[z, y, x], h_xy[z, y, x], h_xx[z, y, x]],
                ]
            )

            hessian_det = linalg.det(hessian_matrix, check_finite=False)
            if hessian_det < 0 and not self.is_negative_definite(hessian_matrix):
                continue
            
            if self.filter_type == 'yang':
                score = self._yang_tubularity(hessian_matrix, point, sigma)
            elif self.filter_type == 'kumar':
                score = self._kumar_vesselness(hessian_matrix, point, sigma)
            else:
                score = self._frangi_vesselness(hessian_matrix, point, sigma)

            filtered_image[point] = score

        return img_as_float(filtered_image)

    def multiscale_filtering(self, volume: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Applies the anisotropic filter across a range of scales (sigmas).

        The final response for each voxel is the maximum response found across
        all scales, allowing detection of neurites with varying thicknesses.

        Args:
            volume (np.ndarray, optional): The input volume. Defaults to class volume.

        Returns:
            np.ndarray: The maximum response volume from multi-scale filtering.
        """
        if volume is None:
            volume = self.volume.copy()

        max_response_volume = np.zeros_like(volume, dtype=float)
        self.logger.info(f"OP_{self.dataset_number}: Starting Multiscale Filtering with {self.filter_type}'s approach.")
        self.logger.info(f" params: sigma_scale: {self.sigmas}, neuron_thresh: {self.neuron_threshold}")

        for sig in self.sigmas:
            self.logger.info(f"OP_{self.dataset_number} -> Filtering at scale: {sig}")
            current_response = self.tubular_filter(volume, sig)
            max_response_volume = np.maximum(max_response_volume, current_response)

        self.logger.info(f"OP_{self.dataset_number}: Multiscale Filtering complete.")
        return img_as_float(max_response_volume)

    def adaptive_mean_mask(
        self,
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
            volume (np.ndarray): The input volume to be thresholded.
            zero_t (bool): If True, simply thresholds at 0.
            tol (float): Convergence tolerance for the threshold.
            max_iterations (int): Maximum number of iterations.

        Returns:
            tuple: A tuple containing the binary boolean mask and the final threshold.
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

    def _compute_eigenvalues(
        self,
        hessian_matrix: np.ndarray,
        point: Tuple[int, int, int],
        sigma: float,
        sort_by: str = "value",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Computes eigenvalues and eigenvectors of the Hessian matrix.

        Eigenvalues are sorted by value in descending order. For a bright tubular
        structure, λ1 is near zero, while λ2 and λ3 are large negative values.
        """
        try:
            eigenvalues, eigenvectors = linalg.eigh(
                hessian_matrix, check_finite=False, lower=False
            )
        except linalg.LinAlgError as e:
            self.logger.warning(
                f"OP_{self.dataset_number}: Eigenvalue decomposition failed for point {point}, sigma {sigma}: {e}"
            )
            return None, None

        if sort_by == "abs":
            sort_indices = np.argsort(np.abs(eigenvalues))
        else:
            sort_indices = np.argsort(eigenvalues)[::-1]
        return eigenvalues[sort_indices], eigenvectors[:, sort_indices]

    def volume_segmentation(
        self, mask: np.ndarray, volume: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Applies a binary mask to the volume, setting non-masked voxels to 0."""
        mask = img_as_bool(mask)
        target_volume = self.volume.copy() if volume is None else volume.copy()

        if mask.shape != target_volume.shape:
            raise ValueError("Mask and volume must have the same shape.")

        target_volume[~mask] = 0
        return img_as_float(target_volume)

    def morphological_denoising(
        self, neuron_mask: np.ndarray, structure: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Removes salt-and-pepper noise using morphological opening and closing."""
        strel = ndi.generate_binary_structure(3, 1) if structure is None else structure

        mask = img_as_bool(
            ndi.binary_closing(
                ndi .binary_opening(neuron_mask, structure=strel), structure=strel
            )
        )
        return mask

    def gradient_magnitude(self, volume: np.ndarray) -> np.ndarray:
        """Computes the gradient magnitude of the volume using Sobel operators."""
        sobel_z = ndi.sobel(volume, 0)
        sobel_y = ndi.sobel(volume, 1)
        sobel_x = ndi.sobel(volume, 2)
        magnitude = np.sqrt(
            np.square(sobel_z) + np.square(sobel_y) + np.square(sobel_x)
        )
        grad_magnitude_max = magnitude.max()
        if grad_magnitude_max != 0:
            magnitude /= grad_magnitude_max
        return magnitude

    def boundary_voxels(self, volume: np.ndarray) -> np.ndarray:
        """Finds the boundary voxels of a binary mask using morphological erosion."""
        mask = volume.astype(bool)
        struct_element = ndi.generate_binary_structure(rank=3, connectivity=3)
        eroded_mask = ndi.binary_erosion(mask, structure=struct_element)
        return mask ^ eroded_mask

    def correct_and_update_root(
        self,
        skeleton_image: np.ndarray,
        original_root: Optional[Tuple[int, int, int]] = None,
    ) -> Optional[Tuple[int, int, int]]:
        """
        Validates if the root is on the skeleton. If not, finds the closest point.

        Args:
            skeleton_image (np.ndarray): The binary skeleton image.
            original_root (tuple, optional): The root point to check.

        Returns:
            A tuple with the new valid root coordinates or None if skeleton is empty.
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
        new_valid_root = tuple(skeleton_voxels[closest_voxel_index].astype(float))

        self.seed_point = new_valid_root
        self.logger.info(f"OP_{self.dataset_number}: Root updated to: {new_valid_root}")
        return new_valid_root

    def pressure_field(self, mask: np.ndarray, metric: str = "euclidean") -> np.ndarray:
        """
        Computes the 'pressure' field (distance transform) for a given mask.
        """
        neuron_mask = img_as_bool(mask)
        if metric == "euclidean":
            return ndi.distance_transform_edt(neuron_mask)
        elif metric == "taxicab":
            return ndi.distance_transform_cdt(neuron_mask, metric="taxicab")

    def thrust_field(
        self, mask: np.ndarray, seed_point: Tuple[int, int, int] = None
    ) -> np.ndarray:
        """
        Computes the 'thrust' field (Euclidean distance from a seed) within a mask.
        """
        if seed_point is None:
            seed_point = self.seed_point

        seed_img = np.zeros_like(mask, dtype=bool)
        seed_img[seed_point] = True

        thrust_field = ndi.distance_transform_edt(~seed_img)
        thrust_field[~mask] = 0
        return thrust_field

    def find_thrust_maxima(
        self, thrust_field: np.ndarray, neuron_mask: np.ndarray, order: int = 1
    ) -> np.ndarray:
        """Finds local maxima in the thrust field."""
        size = 1 + 2 * order
        footprint = np.ones((size, size, size))
        local_max = ndi.maximum_filter(thrust_field, footprint=footprint)

        # A voxel is a local maximum if it matches the maximum-filtered value
        # and is part of the foreground.
        maxima_mask = (thrust_field == local_max) & neuron_mask
        return np.argwhere(maxima_mask)

    def _get_26_neighborhood(
        self, voxel: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int]]:
        """Gets the 26-connected neighbors of a voxel within volume bounds."""
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

    def generate_skel_from_seed(
        self,
        maximas_set: np.ndarray,
        seed_point: Tuple[int, int, int],
        pressure_field: np.ndarray,
        neuron_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Generates the skeleton using a single Dijkstra search from the seed point
        to all terminal points (maxima).

        Args:
            maximas_set (np.ndarray): Coordinates of the terminal (maxima) points.
            seed_point (Tuple[int, int, int]): The coordinates of the seed point.
            pressure_field (np.ndarray): The pressure distance field.
            neuron_mask (np.ndarray): The binary mask of the neuron region.

        Returns:
            np.ndarray: An array of coordinates representing the complete skeleton.
        """
        skeleton_set: Set[Tuple[int, int, int]] = {seed_point}
        delta = 1e-6  # Add delta to avoid division by zero

        self.logger.info(f"OP_{self.dataset_number}: Starting skeleton generation with Dijkstra's search.")

        # Data structures for Dijkstra's algorithm from the seed
        distances = {seed_point: 0}
        previous_nodes = {}
        pq = [(0, seed_point)]  # Priority queue initialized with the seed point

        while pq:
            current_cost, current_voxel = heapq.heappop(pq)

            if current_cost > distances.get(current_voxel, float("inf")):
                continue

            # Explore neighbors
            for neighbor in self._get_26_neighborhood(current_voxel):
                if neuron_mask[neighbor]:
                    # Cost is the inverse of pressure to favor paths through the neurite center
                    edge_weight = 1.0 / (pressure_field[neighbor] + delta)
                    new_cost = current_cost + edge_weight

                    if new_cost < distances.get(neighbor, float("inf")):
                        distances[neighbor] = new_cost
                        previous_nodes[neighbor] = current_voxel
                        heapq.heappush(pq, (new_cost, neighbor))

        self.logger.info(f"OP_{self.dataset_number}: Dijkstra's search complete. Reconstructing paths...")

        # Reconstruct the path from each maximum back to the seed
        for maxima_point in maximas_set:
            current = tuple(maxima_point.astype(int))

            if current not in distances:
                # Skip maxima that were not reached by the search
                continue

            # Trace the path from the maximum back to the seed using predecessors
            while current in previous_nodes:
                skeleton_set.add(current)
                current = previous_nodes.get(current)

        self.logger.info(f"OP_{self.dataset_number}: Branch reconstruction complete.")
        if not skeleton_set:
            return np.array([], dtype=int)

        return np.array(list(skeleton_set), dtype=int)

    @staticmethod
    def is_negative_definite(matrix):
        """
        Checks if a symmetric matrix is negative definite using Cholesky decomposition.

        Args:
            matrix (np.ndarray): The symmetric matrix to check.

        Returns:
            bool: True if the matrix is negative definite, False otherwise.
        """
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Input matrix must be symmetric.")

        # A matrix is negative definite if and only if its negation is positive definite.
        negative_matrix = -matrix

        try:
            # Attempt Cholesky decomposition. It succeeds only for positive-definite matrices.
            np.linalg.cholesky(negative_matrix)
            return True
        except np.linalg.LinAlgError:
            # Decomposition failed, so the matrix is not positive-definite.
            return False
