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


def is_negative_definite(matrix: np.ndarray) -> bool:
    """
    Checks if a symmetric matrix is negative definite using Cholesky decomposition.

    Args:
        matrix: The symmetric matrix to check.

    Returns:
        True if the matrix is negative definite, False otherwise.
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
        return False


def yang_tubularity(
    hessian: np.ndarray,
    sigma: float,
    neuron_threshold: float = 0.05,
) -> float:
    """
    Calculates the tubularity measure based on Hessian eigenvalues (Yang et al. 2013).

    This implements the vesselness function f(u) from Yang et al. (2013) to
    enhance tube-like structures. The function is high when one eigenvalue
    is close to zero and the other two are large and negative.

    Args:
        hessian: The 3x3 Hessian matrix.
        sigma: The scale at which the Hessian was computed.
        neuron_threshold: Threshold to filter out spurious responses.

    Returns:
        The tubularity score (0 if conditions not met).
    """
    eigenvalues, _ = compute_hessian_eigenvalues(hessian, sort_by="value")
    if eigenvalues is None:
        return 0.0

    lambda1, lambda2, lambda3 = eigenvalues

    # Condition for bright, line-like structures:
    # lambda1 is near 0, while lambda2 and lambda3 are negative.
    if lambda2 >= 0 or lambda3 >= 0 or np.abs(lambda1) > neuron_threshold:
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


def frangi_vesselness(
    hessian: np.ndarray,
    sigma: float,
    alpha: float = 0.5,
    beta: float = 0.5,
    c: float = 0,
) -> float:
    """
    Calculates the vesselness response using the Frangi et al. (1998) filter.

    Args:
        hessian: The 3x3 Hessian matrix.
        sigma: The scale at which the Hessian was computed.
        alpha: Sensitivity to plate-like vs line-like structures.
        beta: Sensitivity to blob-like vs line-like structures.
        c: Sensitivity to background noise.

    Returns:
        The vesselness score (0 if conditions not met).
    """
    eigenvalues, _ = compute_hessian_eigenvalues(hessian, sort_by="abs")
    if eigenvalues is None:
        return 0.0

    # For bright tubular structures on a dark background, the two largest
    # absolute eigenvalues (corresponding to eigenvalues[1] and eigenvalues[2])
    # must be negative.
    if eigenvalues[1] >= 0 or eigenvalues[2] >= 0:
        return 0.0

    lambda1_abs, lambda2_abs, lambda3_abs = np.abs(eigenvalues)
    epsilon = 1e-10

    # Define Frangi filter terms
    # Ratio to distinguish between plate-like and line-like structures
    Ra = lambda2_abs / (lambda3_abs + epsilon)
    # Ratio to distinguish between blob-like and line-like structures
    Rb = lambda1_abs / (np.sqrt(lambda2_abs * lambda3_abs) + epsilon)
    # Norm of the Hessian matrix to account for overall structure intensity
    S = np.sqrt(np.sum(np.square(eigenvalues)))
    if c == 0:
        c = 0.5 * S.max()
        # Se S.max for zero (imagem vazia), evita erro
        if c == 0:
            c = 1.0
    # Calculate the filter response
    exp_Ra = 1.0 - np.exp(-(Ra**2) / (2 * alpha**2))
    exp_Rb = np.exp(-(Rb**2) / (2 * beta**2))
    exp_S = 1.0 - np.exp(-(S**2) / (2 * c**2))

    # The vesselness is the product of these three terms
    vesselness = exp_Ra * exp_Rb * exp_S

    return vesselness


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


def sato_tubularity(
    hessian: np.ndarray,
    sigma: float,
    alpha1: float = 0.5,
    alpha2: float = 2.0,
) -> float:
    """
    Calculates tubularity using Sato et al. line filter.

    Args:
        hessian: The 3x3 Hessian matrix.
        sigma: The scale at which the Hessian was computed.
        alpha1: First sensitivity parameter.
        alpha2: Second sensitivity parameter.

    Returns:
        The tubularity score.
    """
    eigenvalues, _ = compute_hessian_eigenvalues(hessian, sort_by="value")
    if eigenvalues is None:
        return 0.0

    lambda1, lambda2, lambda3 = eigenvalues

    # For bright lines: lambda1 ≈ 0, lambda2 < 0, lambda3 < 0
    if lambda2 >= 0 or lambda3 >= 0:
        return 0.0

    # Sato's line filter
    if lambda2 < lambda3 * alpha2:
        return 0.0

    return abs(lambda3) * (1 - np.exp(-((lambda2 / (alpha1 * lambda3)) ** 2)))


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
    # Em vasos brilhantes, a curvatura ortogonal (e2, e3) deve ser negativa [cite: 79]
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

    # Se o threshold for muito restritivo na entrada, use volume > 0
    # Aqui usamos uma máscara para vetorizar apenas o necessário
    mask = (volume > neuron_threshold) & (trace < 0)

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

    if filter_type == "frangi" or filter_type == "yang":  # Assumindo Frangi como base
        # Nota: Yang tem logica diferente, mas aqui focamos na fidelidade ao Frangi
        results = frangi_vesselness_vectorized(eigenvalues, alpha=0.5, beta=0.5)

    # 6. Mapear resultados de volta para o volume 3D
    output[mask] = results

    return output
