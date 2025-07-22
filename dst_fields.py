import numpy as np
from scipy import ndimage as ndi
from scipy import linalg
import logging
from skimage.util import img_as_float, img_as_bool

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
        volume,
        sigma_range=(1, 4, 1),
        step_size=1.0,
        neuron_threshold=1e-2,
    ):
        """
        Initializes the DistanceFields class.

        Args:
            volume (np.ndarray): The 3D input image data.
            sigma_range (tuple): A tuple (min, max, step) for the sigma values used in multi-scale filtering.
            step_size (float): The step size for tracing algorithms.
            neuron_threshold (float): Initial threshold for neuron segmentation.
        """
        self.volume = img_as_float(volume)
        self.shape = volume.shape
        sigma_min, sigma_max, sigma_step = sigma_range
        self.sigmas = np.arange(sigma_min, sigma_max + sigma_step, sigma_step).astype(float)
        if not self.sigmas.size:
            self.sigmas = np.array([sigma_min])

        self.step_size = step_size
        self.neuron_threshold = neuron_threshold

        self.skeleton = dict()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def compute_eigenvalues(self, hessian_matrix, point, sigma):
        """
        Computes eigenvalues and eigenvectors of the Hessian matrix at a point.

        The eigenvalues are sorted by their absolute values in ascending order:
        |λ1| ≤ |λ2| ≤ |λ3|. For a tubular structure, λ1 corresponds to the
        direction of the vessel, while λ2 and λ3 correspond to the cross-section.

        Args:
            hessian_matrix (np.ndarray): The 3x3 Hessian matrix.
            point (tuple): The (z, y, x) coordinates of the voxel.
            sigma (float): The sigma scale at which the Hessian was computed.

        Returns:
            tuple: A tuple containing the sorted eigenvalues (np.ndarray) and
                   sorted eigenvectors (np.ndarray), or (None, None) if
                   decomposition fails.
        """
        if hessian_matrix is None:
            self.logger.debug(f"Hessian matrix is None for point {point} at sigma {sigma}.")
            return None, None

        try:
            eigenvalues, eigenvectors = linalg.eigh(hessian_matrix, check_finite=False)
        except linalg.LinAlgError as e:
            self.logger.warning(f"Eigenvalue decomposition failed for point {point}, sigma {sigma}: {e}")
            return None, None

        # Sort eigenvalues by their absolute values
        sort_indices = np.argsort(np.abs(eigenvalues))
        sorted_eigenvalues = eigenvalues[sort_indices]
        sorted_eigenvectors = eigenvectors[:, sort_indices]

        return sorted_eigenvalues, sorted_eigenvectors

    def adaptive_mean_mask(self, volume, zero_t=False, tol=1e-3, max_iterations=100):
        """
        Generates a binary mask using an iterative adaptive mean thresholding.

        This method determines a global threshold by iteratively averaging the
        mean intensities of pixels above and below the current threshold, as
        described by Yang et al. (2013). The iteration stops when the
        threshold value converges.

        Args:
            volume (np.ndarray): The input volume to be thresholded.

        Returns:
            tuple: A tuple containing:
                   - final_mask (np.ndarray): The resulting binary boolean mask.
                   - current_threshold (float): The converged threshold value.
        """
        
        if zero_t:
            return volume > 0, 0
        else:
            current_threshold = np.mean(volume)

            for _ in range(max_iterations):
                lower_intensity_pixels = volume[volume <= current_threshold]
                higher_intensity_pixels = volume[volume > current_threshold]

                if lower_intensity_pixels.size == 0 or higher_intensity_pixels.size == 0:
                    break

                lower_mean = np.mean(lower_intensity_pixels)
                higher_mean = np.mean(higher_intensity_pixels)
                new_threshold = (lower_mean + higher_mean) / 2

                if abs(new_threshold - current_threshold) < tol:
                    break

                current_threshold = new_threshold

            final_mask = volume > current_threshold
            return final_mask, current_threshold

    def volume_segmentation(self, mask, volume=None):
        """
        Segments a volume by applying a binary mask.

        Sets all voxels in the volume to zero where the corresponding
        mask value is False.

        Args:
            mask (np.ndarray): The binary boolean mask.
            volume (np.ndarray, optional): The volume to segment. If None, the
                                           class's volume is used. Defaults to None.

        Returns:
            np.ndarray: The segmented volume as a float array.
        """
        mask = img_as_bool(mask)
        
        if volume is None:
            volume = self.volume.copy()
            
        if mask.shape != volume.shape:
            raise ValueError("Mask and volume must have the same shape.")
        
        volume[np.logical_not(mask)] = 0
        
        return img_as_float(volume)

    def gradient_magnitude(self, volume):
        """
        Computes the gradient magnitude of the volume using Sobel operators.

        Args:
            volume (np.ndarray): The input 3D image volume.

        Returns:
            np.ndarray: A volume where each voxel's value is the gradient magnitude.
        """
        sobel_z = ndi.sobel(volume, 0)
        sobel_y = ndi.sobel(volume, 1)
        sobel_x = ndi.sobel(volume, 2)

        edge_map = np.sqrt(np.square(sobel_z) + np.square(sobel_y) + np.square(sobel_x))
        return edge_map

    def tubular_enhancer(self, hessian, point, sigma):
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
        eigenvalues, _ = self.compute_eigenvalues(hessian, point, sigma)
        if eigenvalues is None:
            return 0.0

        # Coefficients from the paper: α1=0.5, α2=0.5, α3=25
        alphas = np.array([0.5, 0.5, 25.0])
        sum_lambda_sq = np.sum(np.square(eigenvalues))
        if sum_lambda_sq == 0:
            return 0.0


        lambda1, lambda2, lambda3 = eigenvalues
        # Condition for tubular structure: λ1 ≈ 0 and |λ1| << |λ2|, |λ3|
        is_tubular = (
            np.abs(lambda1) <= self.neuron_threshold and
            np.abs(lambda1) < np.abs(lambda2)  and
            np.abs(lambda1) < np.abs(lambda3) 
        )

        if is_tubular:
            k_factors = np.exp(np.negative((np.square(eigenvalues))) / sum_lambda_sq)

            return np.sum(alphas * k_factors)
        else:
            return 0.0

    def anisotropic_filter(self, volume=None, sigma=None):
        """
        Applies a Hessian-based anisotropic filter to enhance tubular structures.

        This filter implements the equation v = exp(-||∇u||²) * f(u) from the paper,
        where f(u) is the tubularity score from `tubular_enhancer`. It enhances
        line-like structures while suppressing noise and other shapes.

        Args:
            volume (np.ndarray, optional): The input volume. Defaults to class volume.
            sigma (float, optional): The scale (sigma) for the Gaussian derivatives.
                                     Defaults to the minimum sigma of the class.

        Returns:
            np.ndarray: The filtered image as a float array.
        """
        if volume is None:
            volume = self.volume
        if sigma is None:
            sigma = self.sigmas[0]

        filtered_image = np.zeros_like(volume, dtype=float)
        gradient_mag = self.gradient_magnitude(volume)
        term1 = np.exp(-np.square(gradient_mag))

        # Compute second-order derivatives (Hessian components)
        h_xx = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 0, 2], mode='reflect')
        h_yy = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 2, 0], mode='reflect')
        h_zz = ndi.gaussian_filter(volume, sigma=sigma, order=[2, 0, 0], mode='reflect')
        h_xy = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 1, 1], mode='reflect')
        h_xz = ndi.gaussian_filter(volume, sigma=sigma, order=[1, 0, 1], mode='reflect')
        h_yz = ndi.gaussian_filter(volume, sigma=sigma, order=[1, 1, 0], mode='reflect')

        non_zero_voxels = np.nonzero(volume)
        for z, y, x in zip(*non_zero_voxels):
            point = (z, y, x)
            hessian_matrix = np.array([
                [h_zz[z, y, x], h_yz[z, y, x], h_xz[z, y, x]],
                [h_yz[z, y, x], h_yy[z, y, x], h_xy[z, y, x]],
                [h_xz[z, y, x], h_xy[z, y, x], h_xx[z, y, x]]
            ])

            tubularity_score = self.tubular_enhancer(hessian_matrix, point, sigma)
            filtered_image[point] = term1[point] * tubularity_score

        return img_as_float(filtered_image)

    def multiscale_anisotropic(self, volume=None):
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
            volume = self.volume

        max_response_volume = np.zeros_like(volume, dtype=float)
        self.logger.info("Starting Multiscale Anisotropic Filtering...")

        for sig in self.sigmas:
            self.logger.info(f"-> Filtering at scale: {sig}")
            current_response = self.anisotropic_filter(volume, sig)
            max_response_volume = np.maximum(max_response_volume, current_response)
        
        self.logger.info("Multiscale Anisotropic Filtering complete.")
        return img_as_float(max_response_volume)

    def boundary_voxels(self, volume):
        """
        Finds the boundary voxels of a binary mask using morphological erosion.

        A voxel is on the boundary if it is part of the foreground but has at
        least one neighbor in the background. This is achieved efficiently by
        subtracting the eroded mask from the original mask.

        Args:
            volume (np.ndarray): The input binary mask.

        Returns:
            np.ndarray: A boolean mask containing only the boundary voxels.
        """
        mask = volume.astype(bool)

        # 3x3x3 structuring element for 26-pixel connectivity in 3D
        struct_element = ndi.generate_binary_structure(rank=3, connectivity=3)

        eroded_mask = ndi.binary_erosion(mask, structure=struct_element)

        # The boundary is the set difference between the original mask and the eroded mask
        boundary = mask ^ eroded_mask
        return boundary

    def pressure_field(self, mask, metric='euclidean'):
        """
        Computes the 'pressure' field for a given neuron mask.

        The pressure field is defined as the Euclidean distance transform of the
        foreground (neuron mask) to the nearest background pixel. The skeleton
        of the neuron will form ridges of local maxima in this field.

        Args:
            mask (np.ndarray): A binary mask where True represents the neuron.

        Returns:
            np.ndarray: The pressure field, where each voxel value is the
                        distance to the nearest background voxel.
        """
        neuron_mask = img_as_bool(mask)
        
        if metric == 'euclidean': 
            return ndi.distance_transform_edt(neuron_mask)
        elif metric == 'taxicab':
            return ndi.distance_transform_cdt(neuron_mask, metric='taxicab')
        else:
            return ndi.distance_transform_cdt(neuron_mask, metric='chessboard')

    def thrust_field(self, mask, seed_point):
        """
        Computes the 'thrust' field for a given neuron mask and seed point.

        The thrust field is defined as the Euclidean distance transform of the
        foreground pixels to a specified seed point. Terminal points of the
        neuron will appear as local maxima in this field.

        Args:
            mask (np.ndarray): A binary mask where True represents the neuron.
            seed_point (tuple): The (z, y, x) coordinates of the seed point.

        Returns:
            np.ndarray: The computed thrust field.
        """
        seed_img = np.zeros_like(mask, dtype=bool)
        seed_img[seed_point] = True
        
        thrust_field = ndi.distance_transform_cdt(np.logical_not(seed_img), metric='taxicab')
        thrust_field[np.logical_not(mask)] = 0
        
        return thrust_field 
    
    def get_26_neighborhood(self, voxel):
        z, y, x = map(int, voxel)
        neighbors = []

        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nz, ny, nx = z + dz, y + dy, x + dx

                    if (
                        0 <= nz < self.shape[0] and 
                        0 <= ny < self.shape[1] and 
                        0 <= nx < self.shape[2]
                    ):
                        neighbors.append((nz, ny, nx))

        return neighbors
    
    def highest_pressure_neighbor(self, voxel, pressure_field, neuron_mask, visited_points):
        better_neighbor = None 
        highest_pressure = -1.0 
        
        neigbors = self.get_26_neighborhood(voxel)
        neigbors_not_visited = list(filter(lambda x: x not in visited_points, neigbors))
        for neigh in neigbors_not_visited:
            if neuron_mask[neigh]:
                if pressure_field[neigh] > highest_pressure:
                    better_neighbor = neigh
                    highest_pressure = pressure_field[neigh]
                    
        return better_neighbor
    
    def generate_skel_from_maximas(self, maximas_set, seed_point, pressure_field, thrust_field, neuron_mask, epsilon=1.0, max_iter=1000):
        branches = dict()
        for maxima_point in maximas_set:
            maxima_t = tuple(maxima_point.astype(int))
            branch_path = []
            branch_path.append(maxima_t)
            current_point = maxima_t
            visited_points = set()
            for _ in range(max_iter):
                if (current_point[0] == seed_point[0] and
                    current_point[1] == seed_point[1] and
                    current_point[2] == seed_point[2]):
                    self.logger.info(f'branch[{maxima_t}] reached the seed_point({seed_point})')
                    break
                
                visited_points.add(current_point)
                q_star = self.highest_pressure_neighbor(current_point, pressure_field, neuron_mask, visited_points)
                
                if q_star is None:
                    self.logger.info(f'q_star is None. No valid neighbors')
                    break
                
                if not(thrust_field[current_point] + epsilon > thrust_field[q_star]):
                    #self.logger.info(f'Thrust field value of q_star is greater\nthrust_field[current_point]={thrust_field[current_point] + epsilon}\nthrust_field[q_star]={thrust_field[q_star]}')
                    break
                
                branch_path.append(q_star)
                current_point = q_star
                    
            branches[maxima_t] = branch_path

        self.logger.info("Generating skeleton...")
        skeleton_set = set()
        for branch in branches.values():
            skeleton_set.update(branch)
        
        return np.array(list(skeleton_set)).astype(int)
                
        
            
            
            
            
                