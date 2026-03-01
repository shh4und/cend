"""
Hessian-based tubular filtering methods for 3D vessel/neuron enhancement.

This module provides various filter implementations for enhancing tubular structures
in 3D images based on Hessian eigenvalue analysis.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import linalg
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)


def compute_hessian_eigenvalues(
    hessian_matrix: np.ndarray,
    sort_by: str = "value",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Computes eigenvalues and eigenvectors of the Hessian matrix.

    Args:
        hessian_matrix: 3x3 Hessian matrix.
        sort_by: How to sort eigenvalues - "value" (descending) or "abs" (by absolute value).

    Returns:
        Tuple of (eigenvalues, eigenvectors) or (None, None) if computation fails.

    Note:
        For bright tubular structures, λ1 is near zero, while λ2 and λ3 are large negative values.
    """
    try:
        eigenvalues, eigenvectors = linalg.eigh(hessian_matrix, check_finite=False, lower=False)
    except linalg.LinAlgError as e:
        logger.warning(f"Eigenvalue decomposition failed: {e}")
        return None, None

    if sort_by == "abs":
        sort_indices = np.argsort(np.abs(eigenvalues))
    else:
        sort_indices = np.argsort(eigenvalues)[::-1]

    return eigenvalues[sort_indices], eigenvectors[:, sort_indices]


def kumar_vesselness(
    hessian: np.ndarray,
    sigma: float,
    volume_max: float = 1.0,
    c_factor: float = 0.5,
) -> float:
    """
    Calculates the vesselness response using the Kumar et al. (2013) filter.

    This is based on the Multi-scale Vessel Enhancement Filter (MVEF)
    from the reference paper.

    Args:
        hessian: The 3x3 Hessian matrix.
        sigma: The scale at which the Hessian was computed.
        volume_max: Maximum intensity value in the volume (for normalization).
        c_factor: Structureness sensitivity factor.

    Returns:
        The vesselness score (0 if conditions not met).
    """
    eigenvalues, _ = compute_hessian_eigenvalues(hessian, sort_by="abs")
    if eigenvalues is None:
        return 0.0

    lambda1, lambda2, lambda3 = eigenvalues
    abs_l2, abs_l3 = np.abs(lambda2), np.abs(lambda3)
    epsilon = 1e-10

    term1 = 1.0 - (np.abs(abs_l2 - abs_l3) / (abs_l2 + abs_l3 + epsilon))
    term2 = (2.0 / 3.0) * lambda1 - lambda2 - lambda3
    S_structureness = np.sqrt(np.sum(np.square(eigenvalues)))

    c_const = c_factor * volume_max
    if c_const < epsilon:
        c_const = 0.5

    term3_structureness = 1.0 - np.exp(-(S_structureness) / (2 * c_const**2 + epsilon))

    v_sigma = term1 * term2 * term3_structureness

    if v_sigma <= 0:
        return 0.0
    else:
        k = 0.5
        vesselness = v_sigma * np.exp(k * sigma)
        return vesselness


def compute_hessian_eigenvalues_vectorized(hessian_masked):
    """
    Computa autovalores para um array de matrizes (N, 3, 3) de forma vetorizada.
    Garante ordenação por magnitude absoluta: |lambda1| <= |lambda2| <= |lambda3|
    """
    # linalg.eigh é otimizado para matrizes simétricas (Hermitianas)
    # Retorna autovalores em ordem crescente (com sinal)
    eigenvalues = linalg.eigvalsh(hessian_masked)

    # Ordenação por valor absoluto (Fidelidade ao Frangi )
    # argsort no último eixo
    idx = np.argsort(np.abs(eigenvalues), axis=-1)

    # Reorganiza os autovalores baseados nos índices
    sorted_eigenvalues = np.take_along_axis(eigenvalues, idx, axis=-1)

    return sorted_eigenvalues


def frangi_vesselness_vectorized(eigenvalues, alpha=0.5, beta=0.5, c=None):
    """
    Implementação vetorizada do filtro Frangi.
    Esperado eigenvalues shape: (N, 3) onde N é o número de voxels processados.
    Ordenação esperada: |e1| <= |e2| <= |e3|
    """
    # Separação dos autovalores
    lambda1 = eigenvalues[..., 0]
    lambda2 = eigenvalues[..., 1]
    lambda3 = eigenvalues[..., 2]

    # Filtro de polaridade para vasos BRILHANTES em fundo escuro
    # Em vasos brilhantes, a curvatura ortogonal (e2, e3) deve ser negativa
    # Se e2 ou e3 forem positivos, não é um tubo brilhante.
    condition_mask = (lambda2 < 0) & (lambda3 < 0)

    # Prepara arrays de magnitude
    lambda1_abs = np.abs(lambda1)
    lambda2_abs = np.abs(lambda2)
    lambda3_abs = np.abs(lambda3)

    # Evita divisão por zero
    epsilon = 1e-10

    # Razões Geométricas [cite: 95, 100]
    # Ra: Plate-like vs Line-like (|e2| / |e3|)
    Ra = lambda2_abs / (lambda3_abs + epsilon)

    # Rb: Blob-like vs Line-like (|e1| / sqrt(|e2*e3|))
    Rb = lambda1_abs / (np.sqrt(lambda2_abs * lambda3_abs) + epsilon)

    # S: "Second order structureness" (Norma de Frobenius) [cite: 122]
    S = np.sqrt(lambda1**2 + lambda2**2 + lambda3**2)

    # Definição dinâmica de C (fidelidade ao artigo )
    if c is None:
        c = 0.5 * S.max()
        if c == 0:
            c = 1.0

    # Cálculo da Vesselness [cite: 128]
    exp_Ra = 1.0 - np.exp(-(Ra**2) / (2 * alpha**2))
    exp_Rb = np.exp(-(Rb**2) / (2 * beta**2))
    exp_S = 1.0 - np.exp(-(S**2) / (2 * c**2))

    vesselness = exp_Ra * exp_Rb * exp_S

    # Zera onde a polaridade (sinal dos autovalores) está errada
    vesselness[~condition_mask] = 0

    return vesselness


def yang_tubularity_vectorized(eigenvalues, alphas=[0.5, 0.5, 25.0], neuron_threshold=0.05):
    """
    Implementação vetorizada do filtro Yang (Yang et al. 2013).
    Esperado eigenvalues shape: (N, 3) onde N é o número de voxels processados.
    Ordenação esperada: lambda1 >= lambda2 >= lambda3 (por valor, não magnitude)

    Para estruturas brilhantes tipo linha:
    - lambda1 ≈ 0 (maior valor, próximo de zero)
    - lambda2 < 0 (valor médio negativo)
    - lambda3 < 0 (menor valor, mais negativo)
    """
    # Reordenar eigenvalues por valor (descendente) para Yang
    # compute_hessian_eigenvalues_vectorized ordena por magnitude absoluta,
    # mas Yang precisa de ordenação por valor
    sort_indices = np.argsort(eigenvalues, axis=-1)[..., ::-1]
    eigenvalues_sorted = np.take_along_axis(eigenvalues, sort_indices, axis=-1)

    # Separação dos autovalores
    lambda1 = eigenvalues_sorted[..., 0]
    lambda2 = eigenvalues_sorted[..., 1]
    lambda3 = eigenvalues_sorted[..., 2]

    alphas = np.array(alphas)  # Coefficients from the reference paper

    # Condições para estruturas brilhantes tipo linha:
    # lambda1 próximo de zero, lambda2 e lambda3 negativos
    condition_mask = (lambda2 < 0) & (lambda3 < 0) & (np.abs(lambda1) <= neuron_threshold)

    # Soma dos quadrados dos eigenvalues (por voxel)
    sum_lambda_sq = np.sum(np.square(eigenvalues_sorted), axis=-1)

    # Evita divisão por zero
    epsilon = 1e-10

    # Máscara adicional para sum_lambda_sq != 0
    valid_mask = condition_mask & (sum_lambda_sq > epsilon)

    # Inicializa resultado com zeros
    tubularity = np.zeros(eigenvalues_sorted.shape[0])

    # Calcula apenas para voxels válidos
    if np.any(valid_mask):
        # f(u) from Equation 2 in Yang et al. paper
        # k_factors shape: (N_valid, 3)
        eigenvalues_valid = eigenvalues_sorted[valid_mask]
        sum_lambda_sq_valid = sum_lambda_sq[valid_mask]

        k_factors = np.exp(
            -(np.square(eigenvalues_valid) / (2 * sum_lambda_sq_valid[:, np.newaxis]))
        )

        # Tubularidade: soma ponderada dos k_factors
        tubularity[valid_mask] = np.sum(alphas * k_factors, axis=-1)

    return tubularity


def apply_tubular_filter(
    volume: np.ndarray,
    sigma: float,
    filter_type: str = "yang",  # Mantido para compatibilidade, mas focado no Frangi
    neuron_threshold: float = 0.05,
) -> np.ndarray:

    volume = volume.astype(float)
    output = np.zeros_like(volume)

    # 1. Normalização de Escala (Crucial para Multiscale )
    # Multiplicar derivada de 2ª ordem por sigma^2
    scale_factor = sigma**2

    # 2. Cálculo das Derivadas Gaussianas (Todo o volume, pois convolução é rápida)
    # Hxx, Hyy, Hzz, Hxy, Hxz, Hyz
    Hxx = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 0, 2]) * scale_factor
    Hyy = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 2, 0]) * scale_factor
    Hzz = ndi.gaussian_filter(volume, sigma=sigma, order=[2, 0, 0]) * scale_factor
    Hxy = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 1, 1]) * scale_factor
    Hxz = ndi.gaussian_filter(volume, sigma=sigma, order=[1, 0, 1]) * scale_factor
    Hyz = ndi.gaussian_filter(volume, sigma=sigma, order=[1, 1, 0]) * scale_factor

    # 3. Criação de Máscara de Interesse (Optimization)
    # Em vez de processar o ar, processamos apenas onde há sinal ou onde o Traço indica estrutura.
    # Otimização Geométrica Original: Trace < 0 para estruturas brilhantes.
    trace = Hxx + Hyy + Hzz

    # Aqui usamos uma máscara para vetorizar apenas o necessário

    mask = (volume > 0) & (trace < 0)

    # Se a máscara estiver vazia, retorna tudo zero
    if not np.any(mask):
        return output

    # 4. Extração e Construção da Hessiana (Apenas Pixels Úteis)
    # Shape resultante: (N_pixels_uteis, 3, 3) - MUITO menor que (X,Y,Z,3,3)
    # Isso resolve o problema de memória e velocidade da "Nova Implementação"
    hessian_masked = np.zeros((np.sum(mask), 3, 3))

    hessian_masked[:, 0, 0] = Hzz[mask]
    hessian_masked[:, 1, 1] = Hyy[mask]
    hessian_masked[:, 2, 2] = Hxx[mask]
    hessian_masked[:, 0, 1] = hessian_masked[:, 1, 0] = Hyz[mask]
    hessian_masked[:, 0, 2] = hessian_masked[:, 2, 0] = Hxz[mask]
    hessian_masked[:, 1, 2] = hessian_masked[:, 2, 1] = Hxy[mask]

    # 5. Cálculo Vetorizado (Fidelidade à Matemática Nova)
    eigenvalues = compute_hessian_eigenvalues_vectorized(hessian_masked)

    if filter_type == "frangi":
        results = frangi_vesselness_vectorized(eigenvalues, alpha=0.5, beta=0.5)
    elif filter_type == "yang":
        results = yang_tubularity_vectorized(eigenvalues, neuron_threshold=neuron_threshold)
    else:
        # Default para Frangi se tipo não reconhecido
        results = frangi_vesselness_vectorized(eigenvalues, alpha=0.5, beta=0.5)

    # 6. Mapear resultados de volta para o volume 3D
    output[mask] = results

    return output
