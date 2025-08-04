import numpy as np
from scipy import ndimage as ndi
from scipy import linalg
import logging
import time
import matplotlib.pyplot as plt 
from skimage.util import img_as_float
from skimage.feature import peak_local_max

class VesselCenterlineExtractor:
    def __init__(
        self,
        volume,
        sigma_range=(1, 4, 1),
        step_size=1.0,
        vessel_threshold=0.1,
        search_radius=9, # Raio para busca de pico na seção transversal
        local_hessian_radius_factor=3.5, # Fator para determinar o raio para cálculo da Hessiana local (fator * sigma)
    ):
        """
        Inicializa o extrator de linha central para vasos sanguíneos.

        Parâmetros:
        -----------
        volume : ndarray
            Volume 3D da imagem (ex: angiografia, CT, etc.)
        sigma_range : tuple (min, max, step)
            Intervalo de escalas (sigma) para o filtro MVEF.
        step_size : float
            Tamanho do passo para percorrer ao longo do eixo do vaso.
        vessel_threshold : float
            Limiar de intensidade para determinar se um ponto está dentro de um vaso.
        search_radius : int
            Raio para procurar o ponto central na seção transversal do vaso.
        local_hessian_radius_factor : float
            Fator para determinar o raio do volume local para cálculo da Hessiana.
        """
        self.volume = img_as_float(volume)
        self.sigma_min, self.sigma_max, self.sigma_step = sigma_range
        self.sigmas = np.arange(
            self.sigma_min, self.sigma_max + self.sigma_step, self.sigma_step
        )
        if not self.sigmas.size: # Garante que sigmas não esteja vazio se sigma_min > sigma_max
             self.sigmas = np.array([self.sigma_min])

        self.step_size = step_size
        self.vessel_threshold = vessel_threshold
        self.search_radius = search_radius
        self.local_hessian_radius_factor = local_hessian_radius_factor
        self.all_tree_points = set()
        # Estrutura de Cache
        self.eigenvalue_cache = {}

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)

    def clear_cache(self):
        """Limpa os caches armazenados."""
        self.eigenvalue_cache = {}
        self.logger.info("Cache de autovalores limpo.")

    def local_hessian(self, point, sigma):
        """
        Calcula a matriz Hessiana em um ponto específico usando um patch local da imagem.
        """
        z_global, y_global, x_global = map(int, point)
        radius = int(np.ceil(self.local_hessian_radius_factor * sigma)) + 2 

        z_min = max(0, z_global - radius)
        z_max = min(self.volume.shape[0], z_global + radius + 1)
        y_min = max(0, y_global - radius)
        y_max = min(self.volume.shape[1], y_global + radius + 1)
        x_min = max(0, x_global - radius)
        x_max = min(self.volume.shape[2], x_global + radius + 1)

        local_volume_slice = self.volume[z_min:z_max, y_min:y_max, x_min:x_max]

        min_dim_size = int(np.ceil(2 * sigma * 2)) + 1 
        if local_volume_slice.size == 0 or \
           local_volume_slice.shape[0] < min_dim_size or \
           local_volume_slice.shape[1] < min_dim_size or \
           local_volume_slice.shape[2] < min_dim_size:
            self.logger.debug(
                f"Volume local em {point} com sigma {sigma} e raio {radius} é muito pequeno "
                f"({local_volume_slice.shape}) para cálculo da Hessiana. Dimensão mínima necessária: {min_dim_size}."
            )
            return None 

        H_comp = {}
        H_comp['xx'] = ndi.gaussian_filter(local_volume_slice, sigma=sigma, order=[0, 0, 2], mode='reflect')
        H_comp['yy'] = ndi.gaussian_filter(local_volume_slice, sigma=sigma, order=[0, 2, 0], mode='reflect')
        H_comp['zz'] = ndi.gaussian_filter(local_volume_slice, sigma=sigma, order=[2, 0, 0], mode='reflect')
        H_comp['xy'] = ndi.gaussian_filter(local_volume_slice, sigma=sigma, order=[0, 1, 1], mode='reflect') # d/dy d/dx
        H_comp['xz'] = ndi.gaussian_filter(local_volume_slice, sigma=sigma, order=[1, 0, 1], mode='reflect') # d/dz d/dx
        H_comp['yz'] = ndi.gaussian_filter(local_volume_slice, sigma=sigma, order=[1, 1, 0], mode='reflect') # d/dz d/dy

        lz, ly, lx = z_global - z_min, y_global - y_min, x_global - x_min

        if not (0 <= lz < H_comp['xx'].shape[0] and
                0 <= ly < H_comp['xx'].shape[1] and
                0 <= lx < H_comp['xx'].shape[2]):
            self.logger.warning(
                f"Ponto relativo {(lz, ly, lx)} para global {point} está fora "
                f"do patch local filtrado de forma {H_comp['xx'].shape}. "
                f"Forma original local_volume_slice: {local_volume_slice.shape}, z_min_max=({z_min},{z_max}), etc."
            )
            return None 

        h_zz_val = H_comp['zz'][lz, ly, lx]
        h_yy_val = H_comp['yy'][lz, ly, lx]
        h_xx_val = H_comp['xx'][lz, ly, lx]
        # Para consistência com H_ij = H_ji:
        # H_xy = H_yx, H_xz = H_zx, H_yz = H_zy
        # Ondi.gaussian_filter com order=[0,1,1] é d/dy(d/dx I). Usaremos este para H_xy e H_yx.
        # Ondi.gaussian_filter com order=[1,0,1] é d/dz(d/dx I). Usaremos este para H_xz e H_zx.
        # Ondi.gaussian_filter com order=[1,1,0] é d/dz(d/dy I). Usaremos este para H_yz e H_zy.
        h_yz_val = H_comp['yz'][lz, ly, lx] # Corresponde a I_zy ou I_yz
        h_xz_val = H_comp['xz'][lz, ly, lx] # Corresponde a I_zx ou I_xz
        h_xy_val = H_comp['xy'][lz, ly, lx] # Corresponde a I_yx ou I_xy

        # Matriz Hessiana estruturada para indexação (z,y,x). H_comp['zz'] é H[0,0], etc.
        # H_ij = d^2I / dx_i dx_j. Para coordenadas (z,y,x) = (x_0, x_1, x_2)
        # H_00 = I_zz, H_11 = I_yy, H_22 = I_xx
        # H_01 = I_zy, H_02 = I_zx, H_12 = I_yx
        H_matrix = np.array([
            [h_zz_val, h_yz_val, h_xz_val], # Linha Z: d/dz(d/dz), d/dz(d/dy), d/dz(d/dx)
            [h_yz_val, h_yy_val, h_xy_val], # Linha Y: d/dy(d/dz), d/dy(d/dy), d/dy(d/dx)
            [h_xz_val, h_xy_val, h_xx_val]  # Linha X: d/dx(d/dz), d/dx(d/dy), d/dx(d/dx)
        ])
        return H_matrix

    def compute_eigenvalues(self, point, sigma):
        """
        Calcula (ou recupera do cache) autovalores e autovetores da matriz Hessiana em um ponto.
        Autovalores são ordenados por seus valores absolutos: |λ1| ≤ |λ2| ≤ |λ3|.
        λ1 corresponde à direção do vaso. λ2, λ3 à seção transversal.
        """
        cache_key = (point, sigma)
        if cache_key in self.eigenvalue_cache:
            self.logger.debug(f"Cache HIT para autovalores em {point}, sigma {sigma}")
            return self.eigenvalue_cache[cache_key]
        
        self.logger.debug(f"Cache MISS para autovalores em {point}, sigma {sigma}. Calculando...")
        hessian_matrix = self.local_hessian(point, sigma)

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

        # Armazenar no cache
        self.eigenvalue_cache[cache_key] = (sorted_eigenvalues, sorted_eigenvectors)
        return sorted_eigenvalues, sorted_eigenvectors

    def mvef_response(self, point, sigma, c_factor=0.5): 
        """
        Calcula a resposta do Filtro de Realce de Vasos Multiescala (MVEF) em um ponto.
        Baseado nas Equações 4 e 5 do artigo de referência.
        """
        eigenvalues, _ = self.compute_eigenvalues(point, sigma) # Utilizará o cache

        if eigenvalues is None:
            return 0.0

        lambda1, lambda2, lambda3 = eigenvalues 

        abs_l2 = np.abs(lambda2)
        abs_l3 = np.abs(lambda3)
        epsilon = 1e-10 

        if (abs_l2 + abs_l3) < epsilon:
            term1_circularity = 0.0
        else:
            term1_circularity = 1.0 - (np.abs(abs_l2 - abs_l3) / (abs_l2 + abs_l3 + epsilon))

        term2_structure = (2.0/3.0) * lambda1 - lambda2 - lambda3 

        S_squared = np.sqrt(np.square(lambda1) + np.square(lambda2) + np.square(lambda3))
        
        c_const = c_factor * self.volume.max() 
        if c_const < epsilon: 
            c_const = 0.5 

        term3_structureness_norm = 1.0 - np.exp(- (S_squared) / (2 * c_const**2 + epsilon)) 

        v_sigma = term1_circularity * term2_structure * term3_structureness_norm 

        if v_sigma <= 0: 
            vesselness = 0.0
        else:
            k = 0.5 
            vesselness = v_sigma * np.exp(k * sigma) 
            
        return vesselness

    def get_multiscale_response(self, point):
        """
        Calcula a resposta MVEF em múltiplas escalas e retorna a resposta máxima
        e a escala em que ocorreu.
        """
        if not self.is_inside_volume(point):
            return 0.0, self.sigma_min

        responses = np.zeros(len(self.sigmas))
        for i, sigma_val in enumerate(self.sigmas):
            responses[i] = self.mvef_response(point, sigma_val)

        if responses.size == 0: 
            return 0.0, self.sigma_min

        max_response = np.max(responses)
        best_scale_idx = np.argmax(responses)
        best_scale = self.sigmas[best_scale_idx]

        return max_response, best_scale
    
    

    def _find_multiple_peaks_in_cs(self, cs_center_point, v2_cs, v3_cs, sigma_at_cs,
                                     min_peak_dist_factor=0.5, peak_response_thresh_factor=0.75):
        Pc_arr = np.array(cs_center_point)
        v2_norm = v2_cs / (np.linalg.norm(v2_cs) + 1e-10)
        v3_norm = v3_cs / (np.linalg.norm(v3_cs) + 1e-10)

        grid_size = 2 * self.search_radius + 1
        response_grid = np.full((grid_size, grid_size), -np.inf)
        point_map = {} 
        max_grid_response = -np.inf

        for i_grid, i_offset in enumerate(range(-self.search_radius, self.search_radius + 1)):
            for j_grid, j_offset in enumerate(range(-self.search_radius, self.search_radius + 1)):
                cs_point_float = Pc_arr + i_offset * v2_norm + j_offset * v3_norm
                cs_point_int = tuple(np.round(cs_point_float).astype(int))
                
                if not self.is_inside_volume(cs_point_int):
                    continue

                response = self.mvef_response(cs_point_int, sigma_at_cs)
                response_grid[i_grid, j_grid] = response
                point_map[(i_grid, j_grid)] = cs_point_int
                if response > max_grid_response:
                    max_grid_response = response
        
        if max_grid_response < self.vessel_threshold:
            return []

        threshold_abs = max(self.vessel_threshold * 1.15, max_grid_response * peak_response_thresh_factor)
        min_distance_pixels = max(1, int(self.search_radius * min_peak_dist_factor))
        
        peak_coords_in_grid = peak_local_max(response_grid, 
                                             min_distance=min_distance_pixels, 
                                             threshold_abs=threshold_abs, 
                                             exclude_border=False)

        detected_peaks_info = []
        for r_idx, c_idx in peak_coords_in_grid:
            if (r_idx, c_idx) in point_map:
                peak_3d = point_map[(r_idx, c_idx)]
                response_at_peak = response_grid[r_idx, c_idx]
                detected_peaks_info.append((peak_3d, response_at_peak))
        
        detected_peaks_info.sort(key=lambda x: x[1], reverse=True)
        return detected_peaks_info

    def find_center_in_cross_section(self, center_of_cs_plane, v2, v3, best_sigma_at_Pc=None, max_response_at_Pc=None, radius=None):
        """
        Encontra o ponto com a resposta MVEF máxima no plano da seção transversal.
        O plano da seção transversal é definido pelo ponto P_c e vetores v2, v3.
        O MVEF é calculado usando a escala ótima (sigma) encontrada em P_c.
        """
        Pc = np.array(center_of_cs_plane)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
        v3_norm = v3 / (np.linalg.norm(v3) + 1e-10)

        if best_sigma_at_Pc is None or max_response_at_Pc is None:
            max_response_at_Pc, best_sigma_at_Pc = self.get_multiscale_response(tuple(Pc))

        max_response = -np.inf
        max_response_point = None

        for i in range(-self.search_radius, self.search_radius + 1):
            for j in range(-self.search_radius, self.search_radius + 1):
                current_plane_point_float = Pc + i * v2_norm + j * v3_norm
                current_plane_point_int = tuple(np.round(current_plane_point_float).astype(int))

                if not self.is_inside_volume(current_plane_point_int):
                    continue

                response = self.mvef_response(current_plane_point_int, best_sigma_at_Pc)

                if response > max_response:
                    max_response = response
                    max_response_point = current_plane_point_int
        
        if max_response_point is None:
            self.logger.debug(f"Nenhum centro encontrado na seção transversal de {center_of_cs_plane}")
        
        return max_response_point

    def extract_centerline(self, seed_point, max_steps=1000, at_step_log=100):
        """
        Extrai a linha central do vaso a partir de um ponto semente.
        Segue o método CEVD (Centerline Extraction using Vessel Direction).
        """
        self.logger.info(f"Iniciando extração da linha central a partir da semente: {seed_point}")
        # Opcional: Limpar o cache no início de uma nova extração,
        # se cada extração deve ser independente em termos de cache.
        # self.clear_cache() 
        start_time = time.time()
        centerline = []

        current_P = tuple(np.round(seed_point).astype(int))

        response_P0, sigma_P0 = self.get_multiscale_response(current_P)
        if response_P0 < self.vessel_threshold:
            self.logger.warning(
                f"Ponto semente inicial {current_P} tem resposta {response_P0:.7f} "
                f"abaixo do limiar {self.vessel_threshold}. Parando."
            )
            return centerline

        _, eigenvectors_P0 = self.compute_eigenvalues(current_P, sigma_P0)
        if eigenvectors_P0 is None:
            self.logger.warning(f"Não foi possível calcular autovalores para a semente inicial {current_P}. Parando.")
            return centerline
        
        v2_P0, v3_P0 = eigenvectors_P0[:, 1], eigenvectors_P0[:, 2]

        current_C = self.find_center_in_cross_section(current_P, v2_P0, v3_P0, sigma_P0, response_P0)

        if current_C is None:
            self.logger.warning(f"Não foi possível encontrar o centro inicial C0 a partir da semente P0={current_P}. Usando P0 como C0.")
            current_C = current_P 
            response_C0, _ = self.get_multiscale_response(current_C)
            if response_C0 < self.vessel_threshold:
                self.logger.warning(f"Fallback C0={current_C} (de P0) está abaixo do limiar. Parando.")
                return centerline
        
        centerline.append(current_C)
        self.logger.info(f"Primeiro ponto da linha central C0: {current_C}")

        for step_num in range(max_steps):
            self.logger.debug(f"Passo {step_num + 1}/{max_steps}. Centro atual Ck: {current_C}")

            response_Ck, sigma_Ck = self.get_multiscale_response(current_C)
            if response_Ck < self.vessel_threshold:
                self.logger.info(
                    f"Ponto da linha central Ck={current_C} resposta {response_Ck:.7f} "
                    f"está abaixo do limiar. Terminando."
                )
                break

            _, eigenvectors_Ck = self.compute_eigenvalues(current_C, sigma_Ck)
            if eigenvectors_Ck is None:
                self.logger.info(f"Análise de Eigen falhou em Ck={current_C}. Terminando.")
                break
            
            vessel_direction_at_Ck = eigenvectors_Ck[:, 0].copy()

            if len(centerline) > 1:
                vec_prev_to_curr = np.array(current_C) - np.array(centerline[-2])
                if np.dot(vessel_direction_at_Ck, vec_prev_to_curr) < 0:
                    vessel_direction_at_Ck *= -1 

            next_P_float = np.array(current_C) + self.step_size * vessel_direction_at_Ck 
            next_P = tuple(np.round(next_P_float).astype(int))

            if not self.is_inside_volume(next_P):
                self.logger.info(f"Próximo ponto Pk+1={next_P} está fora do volume. Terminando.")
                break
            
            response_Pk_plus_1, sigma_Pk_plus_1 = self.get_multiscale_response(next_P)
            if response_Pk_plus_1 < self.vessel_threshold:
                self.logger.info(
                    f"Próximo ponto Pk+1={next_P} resposta {response_Pk_plus_1:.7f} "
                    f"está abaixo do limiar. Terminando."
                )
                break

            _, eigenvectors_Pk_plus_1 = self.compute_eigenvalues(next_P, sigma_Pk_plus_1) 
            if eigenvectors_Pk_plus_1 is None:
                self.logger.info(f"Análise de Eigen falhou para Pk+1={next_P}. Terminando.")
                break
            
            v2_Pk_plus_1, v3_Pk_plus_1 = eigenvectors_Pk_plus_1[:, 1], eigenvectors_Pk_plus_1[:, 2]

            next_C = self.find_center_in_cross_section(next_P, v2_Pk_plus_1, v3_Pk_plus_1, sigma_Pk_plus_1, response_Pk_plus_1) 

            if next_C is None:
                self.logger.info(f"Não foi possível encontrar o centro Ck+1 a partir de Pk+1={next_P}. Terminando.")
                break
            
            if np.array_equal(next_C, current_C):
                self.logger.info(f"Novo centro Ck+1={next_C} é o mesmo que Ck={current_C}. Terminando para evitar loop.")
                break

            centerline.append(next_C)
            current_C = next_C
            self.logger.debug(f"Novo ponto da linha central Ck+1: {current_C}")

            if step_num % at_step_log == 0 and step_num > 0: # Log a cada n passos
                 self.logger.info(f"No passo: {step_num}, próximo_C: {next_C}, cache_size: {len(self.eigenvalue_cache)}")

            if step_num == max_steps - 1:
                self.logger.info(f"Atingido o número máximo de passos ({max_steps}).")

        end_time = time.time()
        self.logger.info(
            f"Extração da linha central finalizada. Encontrados {len(centerline)} pontos em "
            f"{(end_time - start_time):.7f} segundos. Tamanho final do cache de autovalores: {len(self.eigenvalue_cache)}"
        )
        return centerline

    def is_inside_volume(self, point):
        """Verifica se um ponto está dentro dos limites do volume da imagem."""
        z, y, x = point
        shape = self.volume.shape
        return 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]

    def detect_bifurcations(self, point, v2, v3, sigma=None, max_response=None, radius=None):
        """
        Detect bifurcations by looking for multiple peaks in the cross-section.
        
        Parameters:
        -----------
        point : tuple
            Current point coordinates (z, y, x)
        v2, v3 : ndarray
            Eigenvectors that define the cross-section plane
        search_radius : int, optional
            Radius to search for peaks (defaults to self.search_radius)
            
        Returns:
        --------
        peak_points : list
            List of detected peak points in the cross-section
        """
        
        # Normalize vectors with respect to physical spacing
        point = np.array(point)
        v2 = v2 / np.linalg.norm(v2)
        v3 = v3 / np.linalg.norm(v3)
        
        if radius is None:
            radius = np.round(self.search_radius*1.5).astype(int)
        # Get best scale for this point
        if sigma is None or max_response is None:
            max_response, sigma = self.get_multiscale_response(tuple(point))
        
        # Create a grid of responses in the cross-section plane
        responses = np.zeros((2*radius+1, 2*radius+1))
        grid_to_world = {}
        
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                # Linear combination of v2 and v3 to get a point in the plane
                offset = i * v2 + j * v3
                grid_point = np.round(point + offset).astype(int)
                
                grid_i, grid_j = i + radius, j + radius
                
                # Check if point is within volume bounds
                if self.is_inside_volume(grid_point):
                    # Compute MVEF response at this point
                    response = self.mvef_response(tuple(grid_point), sigma)
                    responses[grid_i, grid_j] = response
                    grid_to_world[(grid_i, grid_j)] = tuple(grid_point)
        
        # Find local maxima in the response grid
        peak_points = []
        threshold = max_response*0.66
        responses_smoothed = ndi.gaussian_filter(responses, sigma=0.5)

        # Simple peak detection - check all 8 neighbors
        for i in range(1, 2*radius):
            for j in range(1, 2*radius):
                is_peak = (responses_smoothed[i, j] > threshold)
                for ni in [-1, 0, 1]:
                    for nj in [-1, 0, 1]:
                        if ni == 0 and nj == 0:
                            continue
                        is_peak = is_peak and (responses_smoothed[i, j] > responses_smoothed[i+ni, j+nj])
                
                if is_peak and (i, j) in grid_to_world:
            
                    peak_points.append(grid_to_world[(i, j)])
        
        self.logger.debug(f" Found {len(peak_points)} peaks in cross-section at {point}")
        return peak_points



    def extract_vessel_tree(self, seed_point, max_branches=20, max_branch_length_limit=100, min_branch_length_limit=5, check_interval=5):
        """
        Extract the complete vessel tree including bifurcations.
        
        Parameters:
        -----------
        seed_point : tuple
            Initial point coordinates (z, y, x) inside the vessel
        max_branches : int
            Maximum number of branches to extract
        max_branch_length_limit : int
            Maximum steps per branch
        check_interval : int
            Interval of points to check for bifurcations
            
        Returns:
        --------
        vessel_tree : dict
            Dictionary of centerlines with branch IDs as keys
        """
        self.logger.info(f" Extracting vessel tree from seed point {seed_point}")
        start_time = time.time()
        
        # Queue of seed points to process (point, parent_branch_id)
        seed_queue = [(seed_point, None)]
        vessel_tree = {}
        branch_id = 0
        
        # Process queue until empty or max_branches reached
        while seed_queue and branch_id < max_branches:
            current_seed, parent_id = seed_queue.pop(0)
            self.logger.info(f" Processing branch {branch_id}, seed: {current_seed}, parent: {parent_id}")
            
            if branch_id == 0:
                centerline = self.extract_centerline(current_seed, max_steps=500)  
                self.all_tree_points.update(centerline)  
            # Extract centerline from this seed
            else:
                centerline = self.extract_centerline(current_seed, max_steps=max_branch_length_limit)
            
            if len(centerline) >= min_branch_length_limit:  # Ensure we found a valid centerline
                vessel_tree[branch_id] = {
                    'centerline': centerline,
                    'parent': parent_id
                }
                self.all_tree_points.update(centerline)
                # Sample points along centerline to check for bifurcations
                check_points = [centerline[i] for i in range(0, len(centerline), check_interval)]
                self.logger.info(f" Branch {branch_id}: checking {len(check_points)} points for bifurcations")
                
                # For each sampled point, check for bifurcations
                for point in check_points:
                    # Get direction and cross-section vectors
                    max_response, sigma = self.get_multiscale_response(point)
                    _, eigenvectors = self.compute_eigenvalues(point, sigma)
                    v2, v3 = eigenvectors[:, 1], eigenvectors[:, 2]
                    
                    # Detect peaks in cross-section
                    peaks = self.detect_bifurcations(point, v2, v3, sigma, max_response)
                    
                    # If multiple peaks found, add new seeds
                    if len(peaks) > 1:
                        self.logger.info(f" Bifurcation detected at {point} with {len(peaks)} peaks")
                        
                        for peak in peaks:
                            # Skip peaks too close to any tree point
                            if not any(np.linalg.norm(np.array(peak) - np.array(p)) < self.step_size*0.9 for p in self.all_tree_points):
                                self.logger.info(f" Adding new seed at {peak} from branch {branch_id}")
                                seed_queue.append((peak, branch_id))
                
                branch_id += 1
                self.logger.info(f" Completed branch {branch_id-1} with {len(centerline)} points")
            else:
                self.logger.info(f" Failed to extract centerline from seed {current_seed}")
        
        end_time = time.time()
        self.logger.info(f" Vessel tree extraction complete: {branch_id} branches, {(end_time - start_time):.2f}s")
        
        # Clear caches to free memory after extraction
        #self.clear_caches()
        
        return vessel_tree