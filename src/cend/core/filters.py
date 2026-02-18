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
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.util import img_as_float

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


def apply_tubular_filter(
    volume: np.ndarray,
    sigma: float,
    filter_type: str = "yang",
    neuron_threshold: float = 0.05,
) -> np.ndarray:
    """
    Applies a Hessian-based anisotropic filter to enhance tubular structures.

    Args:
        volume: The input 3D volume.
        sigma: The scale for Gaussian derivatives.
        filter_type: Type of filter - "yang", "frangi", "kumar", or "sato".
        neuron_threshold: Threshold parameter (used by Yang filter).

    Returns:
        The filtered image with enhanced tubular structures.
    """
    volume = img_as_float(volume)
    filtered_image = np.zeros_like(volume, dtype=float)
    volume_max = volume.max()

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
        hessian_matrix = np.nan_to_num(hessian_matrix)
        hessian_det = linalg.det(hessian_matrix, check_finite=False)
        if hessian_det < 0 and not is_negative_definite(hessian_matrix):
            continue

        if filter_type == "yang":
            score = yang_tubularity(hessian_matrix, sigma, neuron_threshold)
        elif filter_type == "kumar":
            score = kumar_vesselness(hessian_matrix, sigma, volume_max)
        elif filter_type == "sato":
            score = sato_tubularity(hessian_matrix, sigma)
        else:  # frangi
            score = frangi_vesselness(hessian_matrix, sigma)

        filtered_image[point] = score

    return img_as_float(filtered_image)


def vectorized_frangi_filter(
    volume: np.ndarray,
    sigma: float,
    alpha: float = 1,
    beta: float = 0.8,
    c: float = 0,  # Se None, calcula automaticamente
) -> np.ndarray:

    # 1. Calcular Hessiana e Autovalores para todo o volume (Muito rápido em C)
    # H_elems retorna [Hrr, Hrc, Hcc] para 2D ou [Hxx, Hxy, Hxz, Hyy, Hyz, Hzz] para 3D
    H_elems = hessian_matrix(volume, sigma=sigma, order="rc", use_gaussian_derivatives=False)

    # Calcula autovalores. A função já ordena por magnitude: |l1| <= |l2| <= |l3|
    eigvals = hessian_matrix_eigvals(H_elems)

    # eigvals tem shape (3, Z, Y, X). Vamos separar.
    # Nota: No skimage, a ordem de magnitude é crescente.
    # lambda1 é o menor (direção do tubo), lambda2 e 3 são os maiores (seção transversal)
    lambda1 = eigvals[0]
    lambda2 = eigvals[1]
    lambda3 = eigvals[2]

    # 2. Definição de Constantes e Pré-cálculos
    # Para tubos brilhantes em fundo escuro, λ2 e λ3 devem ser negativos.
    # Como |λ2| e |λ3| são grandes, e eles são negativos, isso significa que
    # os valores reais devem ser < 0.
    # Vamos criar uma máscara para zerar o que não for estrutura brilhante
    # Nota: hessian_matrix_eigvals retorna os valores reais, mas ordenados por abs.

    # Condição: λ2 < 0 e λ3 < 0 (concavidade tubular brilhante)
    # A implementação original do Frangi usa lambda2 e lambda3 ordenados por |abs|.
    # Se ordenado por abs, l2 e l3 são os de maior magnitude.
    weights = (lambda2 < 0) & (lambda3 < 0)

    # Evitar divisão por zero
    epsilon = 1e-10

    lambda1_abs = np.abs(lambda1)
    lambda2_abs = np.abs(lambda2)
    lambda3_abs = np.abs(lambda3)

    # 3. Termos do Frangi
    # Ra: Plate vs Line (Diferença entre as duas maiores curvaturas)
    # Se for linha, l2 ~= l3, então Ra ~= 1. Se for prato, l2 << l3 (ou vice versa dependendo da ordenação)
    # Na ordenação do skimage (|l1| < |l2| < |l3|):
    # Tubo ideal: l1=0, l2=l3.
    # Prato ideal: l1=0, l2=0, l3 high.
    Ra = lambda2_abs / (lambda3_abs + epsilon)

    # Rb: Blob vs Line
    # Tubo: l1=0. Rb = 0.
    # Blob: l1=l2=l3. Rb = 1 (aprox).
    Rb = lambda1_abs / (np.sqrt(lambda2_abs * lambda3_abs) + epsilon)

    # S: Structure (Norma de Frobenius)
    S = np.sqrt(lambda1**2 + lambda2**2 + lambda3**2)

    # Ajuste automático de C se não fornecido
    if c == 0:
        c = 0.5 * S.max()
        # Se S.max for zero (imagem vazia), evita erro
        if c == 0:
            c = 1.0

    # 4. Cálculo da Resposta
    term_a = 1.0 - np.exp(-(Ra**2) / (2 * alpha**2))
    term_b = np.exp(-(Rb**2) / (2 * beta**2))
    term_s = 1.0 - np.exp(-(S**2) / (2 * c**2))

    vesselness = term_a * term_b * term_s

    # Aplicar a restrição de "Brilhante sobre Escuro" (zerar onde curvatura é positiva)
    vesselness[~weights] = 0

    # Zerar valores NaN se houver
    vesselness = np.nan_to_num(vesselness)

    return vesselness
