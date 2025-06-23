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
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)


    def compute_eigenvalues(self, hessian_matrix, point, sigma):
        """
        Calcula (ou recupera do cache) autovalores e autovetores da matriz Hessiana em um ponto.
        Autovalores são ordenados por seus valores absolutos: |λ1| ≤ |λ2| ≤ |λ3|.
        λ1 corresponde à direção do vaso. λ2, λ3 à seção transversal.
        """

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

        return sorted_eigenvalues, sorted_eigenvectors

    def mean_threshold(self, volume):

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

        return edge_map

    def tubular_enhancer(self, hessian, point, sigma):
        epsilon = 1e-2
        eigenvals, _ = self.compute_eigenvalues(hessian, point, sigma)
        if eigenvals is None:
            return 0.0
        
        alphas = np.array([0.5, 0.5, 25.0])
        sum_lambda_sq = np.sum(np.square(eigenvals))
        if sum_lambda_sq == 0:
            return 0.0
        
        
        k_ = np.exp(np.negative((np.square(eigenvals))) / sum_lambda_sq)
        
        lambda1, lambda2, lambda3 = eigenvals
        result = 0.0
        if np.abs(lambda1) <= epsilon and (
            np.abs(lambda1) < np.abs(lambda2) 
            and np.abs(lambda1) < np.abs(lambda3) 
        ):
            for i in range(3):
                
                result += alphas[i] * eigenvals[i] * k_[i]

            return result
        else:
            return 0.0

    def anisotropic_filter(self, volume = None, sigma = None):

        if volume is None:
            volume = self.volume
        
        if sigma is None:
            sigma = self.sigma_min

        img_filtered = np.zeros_like(volume, dtype=float)
        grad_mag = self.gradient_magnitude(volume)
        term1 = np.exp(-np.square(grad_mag))
        
        H_xx = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 0, 2], mode='reflect')
        H_yy = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 2, 0], mode='reflect')
        H_zz = ndi.gaussian_filter(volume, sigma=sigma, order=[2, 0, 0], mode='reflect')
        H_xy = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 1, 1], mode='reflect') # d/dy d/dx
        H_xz = ndi.gaussian_filter(volume, sigma=sigma, order=[1, 0, 1], mode='reflect') # d/dz d/dx
        H_yz = ndi.gaussian_filter(volume, sigma=sigma, order=[1, 1, 0], mode='reflect') # d/dz d/dy
        
        non_zero_voxels = np.nonzero(volume)  # Get indices of all non-zero voxels
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

    def multiscale_anisotropic(self, volume=None):
        if volume is None:
            volume = self.volume
            
        max_volume_response = np.zeros_like(volume, dtype=float)
        self.logger.info(f"Starting Multiscale Anisotropic Filtering...")

        for sig in self.sigmas:
            self.logger.info(f" Multiscale Anisotropic Filtering at {sig} scale")
            curr_response =  self.anisotropic_filter(volume, sig)
            
            max_volume_response = np.maximum(max_volume_response, curr_response)
            
        return img_as_float(max_volume_response)    
            
        