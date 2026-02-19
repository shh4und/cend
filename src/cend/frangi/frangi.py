import numpy as np

from .hessian import absolute_hessian_eigenvalues
from .utils import divide_nonzero


def frangi(
    nd_array,
    scale_range=(1, 2),
    scale_step=0.5,
    alpha=0.5,
    beta=0.5,
    frangi_c=500,
    black_vessels=False,
):

    if nd_array.ndim != 3:
        raise ValueError("Apenas 3 dimensões são suportadas atualmente")

    sigmas = np.arange(scale_range[0], scale_range[1], scale_step)
    if np.any(sigmas < 0.0):
        raise ValueError("Valores de Sigma menores que zero não são válidos")
    print(f"Sigma range: {sigmas}")
    # Inicializa a matriz de saída com zeros do mesmo tamanho da entrada
    # Em vez de criar (N_sigmas, X, Y, Z), criamos apenas (X, Y, Z)
    max_vesselness = np.zeros_like(nd_array, dtype=float)

    for sigma in sigmas:
        # Obtém autovalores ordenados por magnitude absoluta (|e1| <= |e2| <= |e3|)
        eigenvalues = absolute_hessian_eigenvalues(nd_array, sigma=sigma, scale=False)
        print(f"Computing vesselness for sigma value: {sigma}")
        # Calcula vesselness para a escala atual
        current_vesselness = compute_vesselness(
            eigenvalues[0],
            eigenvalues[1],
            eigenvalues[2],
            alpha=alpha,
            beta=beta,
            c=frangi_c,
            black_white=black_vessels,
        )

        # Mantém apenas a resposta máxima entre as escalas (Maximum Intensity Projection across scales)
        # np.maximum é in-place se usarmos o parâmetro 'out', mas aqui a atribuição é segura
        max_vesselness = np.maximum(max_vesselness, current_vesselness)

    return max_vesselness


def compute_measures(eigen1, eigen2, eigen3):
    """
    Ra - estruturas tipo placa (plate-like)
    Rb - estruturas tipo blob (blob-like)
    S - background (norma de Frobenius)
    """
    # No Frangi original, assume-se ordenação |e1| <= |e2| <= |e3|
    # Ra = |e2| / |e3|
    Ra = divide_nonzero(np.abs(eigen2), np.abs(eigen3))

    # Rb = |e1| / sqrt(|e2 * e3|)
    Rb = divide_nonzero(np.abs(eigen1), np.sqrt(np.abs(np.multiply(eigen2, eigen3))))

    # S = sqrt(e1^2 + e2^2 + e3^2)
    S = np.sqrt(np.square(eigen1) + np.square(eigen2) + np.square(eigen3))
    return Ra, Rb, S


def compute_vesselness(eigen1, eigen2, eigen3, alpha, beta, c, black_white):
    Ra, Rb, S = compute_measures(eigen1, eigen2, eigen3)

    plate = 1 - np.exp(-(np.square(Ra)) / (2 * np.square(alpha)))
    blob = np.exp(-(np.square(Rb)) / (2 * np.square(beta)))
    background = 1 - np.exp(-(np.square(S)) / (2 * np.square(c)))

    vesselness = plate * blob * background

    return filter_out_background(vesselness, black_white, eigen2, eigen3)


def filter_out_background(voxel_data, black_white, eigen2, eigen3):
    """
    Zera voxels baseados no sinal dos autovalores.
    Para vasos tubulares em 3D, esperamos lambda2 e lambda3 com sinais específicos.
    """
    # Condições baseadas na geometria do Frangi:
    # Vasos BRILHANTES (em fundo escuro): e2 < 0, e3 < 0
    # Vasos ESCUROS (em fundo claro): e2 > 0, e3 > 0

    if black_white:
        # Se black_white=True, buscamos vasos ESCUROS (dark vessels).
        # A implementação original zerava se e2 < 0 ou e3 < 0.
        # Isso significa que só aceita positivos.
        mask = (eigen2 < 0) | (eigen3 < 0)
    else:
        # Buscamos vasos BRILHANTES. Aceitamos apenas negativos.
        mask = (eigen2 > 0) | (eigen3 > 0)

    voxel_data[mask] = 0

    # Limpeza de NaNs que podem ter surgido nas divisões
    np.nan_to_num(voxel_data, copy=False, nan=0.0)

    return voxel_data
