import numpy as np
from scipy import ndimage as ndi
from scipy import linalg
import logging
import time
from skimage.util import img_as_float, img_as_ubyte


class DistanceFields:

    def __init__(
        self,
        volume,
        sigma_range=(1, 4, 1),
        step_size=1.0,
        neuron_threshold=0.1,
        search_radius=9,
        local_hessian_radius_factor=3.5,
    ):
        self.volume = img_as_float(volume)
        self.sigma_min, self.sigma_max, self.sigma_step = sigma_range
        self.sigmas = np.arange(
            self.sigma_min, self.sigma_max + self.sigma_step, self.sigma_step
        ).astype(float)
        if not self.sigmas.size:
            self.sigmas = np.array([self.sigma_min])

        self.step_size = step_size
        self.neuron_threshold = neuron_threshold
        self.search_radius = search_radius
        self.local_hessian_radius_factor = local_hessian_radius_factor
        self.eigenvalue_cache = {}
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)


    def compute_eigenvalues(self, hessian_matrix, point, sigma):
        """
        Calcula (ou recupera do cache) autovalores e autovetores da matriz Hessiana em um ponto.
        Autovalores são ordenados por seus valores absolutos: |λ1| ≤ |λ2| ≤ |λ3|.
        λ1 corresponde à direção do vaso. λ2, λ3 à seção transversal.
        """
        cache_key = (point, sigma)
        if cache_key in self.eigenvalue_cache:
            self.logger.debug(f"Cache HIT para autovalores em {point}, sigma {sigma}")
            return self.eigenvalue_cache[cache_key]

        if hessian_matrix is None:
            self.logger.debug(f"Matriz Hessiana é None para o ponto {point} em sigma {sigma}.")
            # Não armazenar None no cache para permitir nova tentativa se a condição mudar
            return None, None

        try:
            eigenvalues, eigenvectors = linalg.eigh(hessian_matrix, check_finite=False)
        except linalg.LinAlgError as e:
            self.logger.warning(f"Decomposição de autovalores falhou para o ponto {point}, sigma {sigma}: {e}")
            return None, None

        idx = np.argsort(np.abs(eigenvalues))
        sorted_eigenvalues = eigenvalues[idx]
        sorted_eigenvectors = eigenvectors[:, idx] 

        self.eigenvalue_cache[cache_key] = (sorted_eigenvalues, sorted_eigenvectors)
        return sorted_eigenvalues, sorted_eigenvectors

    def mean_threshold(self):

        volume = self.volume.copy()
        curr_thresh = np.mean(volume)
        while True:

            lower_portion = volume[volume <= curr_thresh]
            higher_portion = volume[volume > curr_thresh]

            if len(lower_portion) == 0 or len(higher_portion) == 0:
                break

            lower_mean = np.mean(lower_portion)
            higher_mean = np.mean(higher_portion)

            new_thresh = (lower_mean + higher_mean) / 2

            if curr_thresh == new_thresh:
                break
            else:
                curr_thresh = new_thresh

        final_vol = volume.copy()
        final_vol[volume <= curr_thresh] = 0

        return (final_vol, curr_thresh)

    def gradient_magnitude(self, volume):
        edge_map = np.zeros_like(volume, dtype=float)

        sobelz = ndi.sobel(volume, 0)
        sobely = ndi.sobel(volume, 1)
        sobelx = ndi.sobel(volume, 2)

        edge_map = np.sqrt(np.square(sobelz) + np.square(sobely) + np.square(sobelx))

        return img_as_float(edge_map)

    def tubular_enhancer(self, hessian, point, sigma):
        epsilon = 1e-3
        eigenvals, _ = self.compute_eigenvalues(hessian, point, sigma)
        if eigenvals is None:
            return 0.0
        
        alphas = np.array([0.5, 0.5, 25.0])
        sum_lambda_sq = np.sum(np.square(eigenvals))
        if sum_lambda_sq == 0:
            return 0.0
        
        
        k_ = np.exp(np.negative((np.square(eigenvals))) / sum_lambda_sq)
        
        lambda1, lambda2, lambda3 = eigenvals
        f_u = 0.0
        if np.abs(lambda1) <= 0 + epsilon and (
            np.abs(lambda1) * 2 < np.abs(lambda2)
            and np.abs(lambda1) * 2 < np.abs(lambda3)
        ):
            for i in range(3):
                
                result += alphas[i] * eigenvals[i] * k_[i]

            return result
        else:
            return 0.0

    def anisotropic_filter(self, sigma = None):

        if sigma == None:
            sigma = self.sigma_min

        img_filtered = np.zeros_like(self.volume, dtype=float)
        grad_mag = self.gradient_magnitude(self.volume)
        term1 = np.exp(-np.square(grad_mag))
        
        H_xx = ndi.gaussian_filter(self.volume, sigma=sigma, order=[0, 0, 2], mode='reflect')
        H_yy = ndi.gaussian_filter(self.volume, sigma=sigma, order=[0, 2, 0], mode='reflect')
        H_zz = ndi.gaussian_filter(self.volume, sigma=sigma, order=[2, 0, 0], mode='reflect')
        H_xy = ndi.gaussian_filter(self.volume, sigma=sigma, order=[0, 1, 1], mode='reflect') # d/dy d/dx
        H_xz = ndi.gaussian_filter(self.volume, sigma=sigma, order=[1, 0, 1], mode='reflect') # d/dz d/dx
        H_yz = ndi.gaussian_filter(self.volume, sigma=sigma, order=[1, 1, 0], mode='reflect') # d/dz d/dy
        
        non_zero_voxels = np.nonzero(self.volume)  # Get indices of all non-zero voxels
        for z, y, x in zip(*non_zero_voxels):
            point = (z, y, x)
            h_zz_val = H_zz[z, y, x]
            h_yy_val = H_yy[z, y, x]
            h_xx_val = H_xx[z, y, x]

            h_xz_val = H_xz[z, y, x] 
            h_xy_val = H_xy[z, y, x] 
            h_yz_val = H_yz[z, y, x] 

            H_matrix = np.array([
                [h_zz_val, h_yz_val, h_xz_val],
                [h_yz_val, h_yy_val, h_xy_val],
                [h_xz_val, h_xy_val, h_xx_val] 
            ])
            
            f_u = self.tubular_enhancer(H_matrix, point, sigma)
            
            img_filtered[point] = term1[point] * f_u
        
        return img_as_float(img_filtered)
