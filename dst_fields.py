import numpy as np
from scipy import ndimage as ndi
from scipy import linalg
import logging
from skimage.util import img_as_float, img_as_bool
from skimage.morphology import remove_small_objects
from typing import Tuple, Optional, Dict, List, Set
import heapq


class DistanceFields:
    """
    Implements the DF-Tracing method for automatic 3D neuron reconstruction.

    This class provides the tools to perform the first two steps of the DF-Tracing
    algorithm as described in "A distance-field based automatic neuron tracing
    method" by Yang et al. (2013). This includes:
    1.  Image preprocessing with multi-scale anisotropic filtering to enhance
        neurite signals.
    2.  Generation of distance fields (pressure field) necessary for
        skeletonization.

    Attributes:
        volume (np.ndarray): The input 3D image volume, converted to float.
        sigmas (np.ndarray): An array of sigma values for multi-scale analysis.
        step_size (float): The step size used in path-finding algorithms.
        neuron_threshold (float): The threshold to segment neuron from background.
    """

    def __init__(
        self,
        volume: np.ndarray,
        sigma_range: Tuple[float, float, float] = (1, 4, 1),
        step_size: float = 1.0,
        neuron_threshold: float = 1e-2,
        seed_point: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """
        Initializes the DistanceFields class.

        Args:
            volume (np.ndarray): The 3D input image data.
            sigma_range (tuple): A tuple (min, max, step) for the sigma values used in multi-scale filtering.
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

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_skeleton(self, skel: np.ndarray):
        self.skeleton = skel

    def get_skeleton(self):
        return self.skeleton

    def _tubular_enhancer(
        self, hessian: np.ndarray, point: Tuple[int, int, int], sigma: float
    ) -> float:
        """
        Calculates the tubularity measure based on Hessian eigenvalues.

        This implements the vesselness function f(u) from Yang et al. (2013),
        which enhances tube-like structures. The function is high when one
        eigenvalue is close to zero and the other two are large and negative,
        indicating a bright tubular structure.

        Args:
            hessian (np.ndarray): The 3x3 Hessian matrix.
            point (tuple): The coordinates of the current voxel.
            sigma (float): The scale at which the Hessian was computed.

        Returns:
            float: The tubularity score, ranging from 0.0 to a positive value.
        """
        eigenvalues = self._compute_eigenvalues(hessian, point, sigma)[0]
        if eigenvalues is None:
            return 0.0

        lambda1, lambda2, lambda3 = eigenvalues

        # Condição para estruturas brilhantes tipo linha (sinal alto)
        # lambda1 proximo de 0, lambda2 e lambda3 negativos.
        if lambda2 >= 0 or lambda3 >= 0 or np.abs(lambda1) > self.neuron_threshold:
            return 0.0

        alphas = np.array([0.5, 0.5, 25.0])  # Coeficientes do artigo
        sum_lambda_sq = np.sum(np.square(eigenvalues))
        if sum_lambda_sq == 0:
            return 0.0

        # f(u) da Equação 2
        k_factors = np.exp(
            -(np.square(eigenvalues) / (2 * sum_lambda_sq))
        )  # O artigo pode ter um erro de digitação, geralmente há um fator 2
        tubularity = np.sum(alphas * k_factors)

        return tubularity

    def anisotropic_filter(
        self, volume: Optional[np.ndarray] = None, sigma: Optional[float] = None
    ) -> np.ndarray:
        """
        Applies a Hessian-based anisotropic filter to enhance tubular structures.

        This filter implements the equation v = exp(-||∇u||²) * f(u) from the paper,
        where f(u) is the tubularity score from `_tubular_enhancer`. It enhances
        line-like structures while suppressing noise and other shapes.

        Args:
            volume (np.ndarray, optional): The input volume. Defaults to class volume.
            sigma (float, optional): The scale (sigma) for the Gaussian derivatives.
                                    Defaults to the minimum sigma of the class.

        Returns:
            np.ndarray: The filtered image as a float array.
        """
        if volume is None:
            volume = self.volume.copy()
        if sigma is None:
            sigma = self.sigmas[0]

        filtered_image = np.zeros_like(volume, dtype=float)
        gradient_mag = self.gradient_magnitude(volume)
        term1 = np.exp(-np.square(gradient_mag))

        # Compute second-order derivatives (Hessian components)
        h_xx = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 0, 2], mode="reflect")
        h_yy = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 2, 0], mode="reflect")
        h_zz = ndi.gaussian_filter(volume, sigma=sigma, order=[2, 0, 0], mode="reflect")
        h_xy = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 1, 1], mode="reflect")
        h_xz = ndi.gaussian_filter(volume, sigma=sigma, order=[1, 0, 1], mode="reflect")
        h_yz = ndi.gaussian_filter(volume, sigma=sigma, order=[1, 1, 0], mode="reflect")

        non_zero_voxels = np.nonzero(volume)
        for z, y, x in zip(*non_zero_voxels):
            point = (z, y, x)
            hessian_matrix = np.array(
                [
                    [h_zz[z, y, x], h_yz[z, y, x], h_xz[z, y, x]],
                    [h_yz[z, y, x], h_yy[z, y, x], h_xy[z, y, x]],
                    [h_xz[z, y, x], h_xy[z, y, x], h_xx[z, y, x]],
                ]
            )

            tubularity_score = self._tubular_enhancer(hessian_matrix, point, sigma)
            filtered_image[point] = term1[point] * tubularity_score

        return img_as_float(filtered_image)

    def multiscale_anisotropic(self, volume: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Applies the anisotropic filter across a range of scales (sigmas).

        The final response for each voxel is the maximum response found across
        all scales. This allows detection of neurites with varying thicknesses.

        Args:
            volume (np.ndarray, optional): The input volume. Defaults to class volume.

        Returns:
            np.ndarray: The maximum response volume from multi-scale filtering.
        """
        if volume is None:
            volume = self.volume.copy()

        max_response_volume = np.zeros_like(volume, dtype=float)
        self.logger.info("Starting Multiscale Anisotropic Filtering...")
        self.logger.info(
            f" params: sigma_scale: {self.sigmas}, neuron_thresh: {self.neuron_threshold}"
        )

        for sig in self.sigmas:
            self.logger.info(f"-> Filtering at scale: {sig}")
            current_response = self.anisotropic_filter(volume, sig)
            max_response_volume = np.maximum(max_response_volume, current_response)

        self.logger.info("Multiscale Anisotropic Filtering complete.")
        return img_as_float(max_response_volume)

    def adaptive_mean_mask(
        self,
        volume: np.ndarray,
        zero_t: bool = False,
        tol: float = 1e-3,
        max_iterations: int = 100,
    ) -> Tuple[np.ndarray, float]:
        """
        Generates a binary mask using an iterative adaptive mean thresholding

        This method determines a global threshold by iteratively averaging the
        mean intensities of pixels above and below the current threshold.

        Args:
            volume (np.ndarray): The input volume to be thresholded.
            zero_t (bool): If True, simply thresholds at 0.
            tol (float): Convergence tolerance for the threshold.
            max_iterations (int): Maximum number of iterations.

        Returns:
            tuple: A tuple containing the binary boolean mask and the converged threshold value.
        """
        if zero_t:
            return volume > 0, 0.0

        current_threshold = np.mean(volume)

        for _ in range(max_iterations):
            # Use boolean masking to avoid creating copies of data subsets
            higher_mask = volume > current_threshold
            lower_mask = ~higher_mask

            # Check if any region is empty
            if not np.any(higher_mask) or not np.any(lower_mask):
                break

            # Calculate means efficiently
            mean_higher = np.sum(volume[higher_mask]) / np.count_nonzero(higher_mask)
            mean_lower = np.sum(volume[lower_mask]) / np.count_nonzero(lower_mask)

            new_threshold = (mean_lower + mean_higher) / 2

            if abs(new_threshold - current_threshold) < tol:
                break

            current_threshold = new_threshold

        final_mask = volume > current_threshold
        return final_mask, current_threshold

    def _compute_eigenvalues(
        self, hessian_matrix: np.ndarray, point: Tuple[int, int, int], sigma: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Computes eigenvalues and eigenvectors of the Hessian matrix at a point.

        The eigenvalues are sorted by their absolute values in ascending order:
        |λ1| ≤ |λ2| ≤ |λ3|. For a tubular structure, λ1 corresponds to the
        direction of the vessel, while λ2 and λ3 correspond to the cross-section.
        """
        try:
            eigenvalues, eigenvectors = linalg.eigh(
                hessian_matrix, check_finite=False, lower=False
            )
        except linalg.LinAlgError as e:
            self.logger.warning(
                f"Eigenvalue decomposition failed for point {point}, sigma {sigma}: {e}"
            )
            return None, None

        sort_indices = np.argsort(eigenvalues)[
            ::-1
        ]  # for bright and tubular structures lambda2 and lambda3 are high-negatives values
        return eigenvalues[sort_indices], eigenvectors[:, sort_indices]

    def volume_segmentation(
        self, mask: np.ndarray, volume: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Segments a volume by applying a binary mask.
        """
        mask = img_as_bool(mask)

        target_volume = self.volume.copy() if volume is None else volume.copy()

        if mask.shape != target_volume.shape:
            raise ValueError("Mask and volume must have the same shape.")

        target_volume[~mask] = 0

        return img_as_float(target_volume)

    def morphological_denoising(
        self, neuron_mask: np.ndarray, structure: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Denoise salt and pepper noise applying opening and closing operetations.
        """
        strel = ndi.generate_binary_structure(3, 1) if structure is None else structure

        mask = img_as_bool(
            ndi.binary_closing(
                ndi.binary_opening(neuron_mask, structure=strel), structure=strel
            )
        )

        return mask

    def gradient_magnitude(self, volume: np.ndarray) -> np.ndarray:
        """
        Computes the gradient magnitude of the volume using Sobel operators.
        """
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
        """
        Finds the boundary voxels of a binary mask using morphological erosion.
        """
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
        Checks if original root is inside skeleton, if not, find the closest one point.

        Args:
            skeleton_image (np.ndarray)
            original_root (tuple)

        Returns:
            tuple or None.
        """

        if not np.any(skeleton_image):
            print("empty skeleton image")
            return None

        root = self.seed_point if original_root is None else original_root

        # 1. Verifica se a raiz original é válida
        if skeleton_image[root]:
            print("original root is inside skeleton")
            return root

        skeleton_voxels = np.argwhere(skeleton_image)

        # square (faster) euclidian distance from all skeleton points to the original root point
        distances_sq = np.sum((skeleton_voxels - np.array(root)) ** 2)

        # find closest index point from original root
        closest_voxel_index = np.argmin(distances_sq)

        new_valid_root = tuple(skeleton_voxels[closest_voxel_index].astype(float))
        self.seed_point = new_valid_root
        print(f"Root updated: {new_valid_root}. Old root: {root}")
        return new_valid_root

    def find_seed_point(
        self, neuron_mask: np.ndarray
    ) -> Optional[Tuple[int, int, int]]:
        """Selects a random point from the boundary voxels."""
        boundary = self.boundary_voxels(neuron_mask)
        boundary_coords = np.argwhere(boundary)
        if boundary_coords.size > 0:
            seed_coord = boundary_coords[np.random.choice(len(boundary_coords))]
            return tuple(seed_coord)
        return None

    def pressure_field(self, mask: np.ndarray, metric: str = "euclidean") -> np.ndarray:
        """
        Computes the 'pressure' field for a given neuron mask.
        """
        neuron_mask = img_as_bool(mask)
        if metric == "euclidean":
            return ndi.distance_transform_edt(neuron_mask)
        elif metric == "taxicab":
            return ndi.distance_transform_cdt(neuron_mask, metric="taxicab")
        else:
            return ndi.distance_transform_cdt(neuron_mask, metric="chessboard")

    def thrust_field(
        self, mask: np.ndarray, seed_point: Tuple[int, int, int] = None
    ) -> np.ndarray:
        """
        Computes the 'thrust' field for a given neuron mask and seed point.
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
        """Finds local maxima from the thrust field."""

        size = 1 + 2 * order
        footprint = np.ones((size, size, size))

        local_max = ndi.maximum_filter(thrust_field, footprint=footprint)

        # a voxel is a local maxima if its value is the same as the maximum filter
        # and belongs to foreground
        maxima_mask = (thrust_field == local_max) & neuron_mask

        return np.argwhere(maxima_mask)

    def _get_26_neighborhood(
        self, voxel: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int]]:
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

    def _highest_pressure_neighbor(
        self,
        voxel: Tuple[int, int, int],
        pressure_field: np.ndarray,
        neuron_mask: np.ndarray,
        visited_points: Set[Tuple[int, int, int]],
    ) -> Optional[Tuple[int, int, int]]:
        better_neighbor: Optional[Tuple[int, int, int]] = None
        highest_pressure = -1.0

        neighbors = self._get_26_neighborhood(voxel)

        for neigh in filter(lambda n: n not in visited_points, neighbors):
            if neuron_mask[neigh] and pressure_field[neigh] > highest_pressure:
                better_neighbor = neigh
                highest_pressure = pressure_field[neigh]

        return better_neighbor

    def generate_skel_from_seed(
        self,
        maximas_set: np.ndarray,
        seed_point: Tuple[int, int, int],
        pressure_field: np.ndarray,
        neuron_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Gera o esqueleto do neurônio usando uma única busca de Dijkstra a partir do
        ponto semente para encontrar os caminhos para todos os pontos terminais.

        Args:
            maximas_set (np.ndarray): Conjunto de coordenadas dos pontos terminais (máximos).
            seed_point (Tuple[int, int, int]): As coordenadas do ponto semente.
            pressure_field (np.ndarray): O campo de distância de pressão.
            neuron_mask (np.ndarray): A máscara binária da região do neurônio.

        Returns:
            np.ndarray: Um array de coordenadas representando o esqueleto completo.
        """
        skeleton_set: Set[Tuple[int, int, int]] = {seed_point}
        delta = 1e-6  # Evitar divisão por zero

        self.logger.info(
            " - Iniciando a geração de esqueleto com busca Dijkstra reversa (1-para-muitos)."
        )

        # Estruturas de dados para a busca única a partir do semente
        distances = {seed_point: 0}
        previous_nodes = {}
        pq = [(0, seed_point)]  # Fila de prioridade começa com o semente

        # A busca expande a partir do semente até a fila esvaziar
        while pq:
            current_cost, current_voxel = heapq.heappop(pq)

            if current_cost > distances.get(current_voxel, float('inf')):
                continue

            # Explorar vizinhos
            for neighbor in self._get_26_neighborhood(current_voxel):
                if neuron_mask[neighbor]:
                    # Custo é o inverso da pressão para favorecer o centro do neurito
                    edge_weight = 1.0 / (pressure_field[neighbor] + delta)
                    new_cost = current_cost + edge_weight

                    if new_cost < distances.get(neighbor, float('inf')):
                        distances[neighbor] = new_cost
                        previous_nodes[neighbor] = current_voxel
                        heapq.heappush(pq, (new_cost, neighbor))

        self.logger.info(" - Busca de Dijkstra concluída. Reconstruindo caminhos...")

        # Agora, para cada máximo, reconstrua o caminho de volta para o semente
        for maxima_point in maximas_set:
            current = tuple(maxima_point.astype(int))
            
            # Se o máximo não foi alcançado pela busca, pule
            if current not in distances:
                # self.logger.warning(f"O ponto máximo {current} não foi alcançado pela busca a partir do semente.")
                continue

            # Traça o caminho do máximo de volta ao semente usando os predecessores
            while current in previous_nodes:
                skeleton_set.add(current)
                current = previous_nodes.get(current)

        self.logger.info(" - Reconstrução de todos os ramos concluída.")
        if not skeleton_set:
            return np.array([], dtype=int)

        return np.array(list(skeleton_set), dtype=int)