import numpy as np
from scipy import linalg


def divide_nonzero(array1, array2, default=1e-10):
    """
    Divide arrays de forma segura, substituindo zeros no denominador.
    Usa np.divide com o argumento 'where' e 'out' para eficiência.
    """
    # Evita alocação de memória desnecessária para cópia do denominador
    return np.divide(array1, array2, out=np.zeros_like(array1, dtype=float), where=array2 != 0)


def absolute_eigenvaluesh(nd_array):
    """
    Calcula autovalores de uma matriz simétrica e os ordena por valor absoluto.
    Input shape esperado: (..., ndim, ndim)
    """
    # eigh é otimizado para matrizes hermitianas/simétricas
    eigenvalues = linalg.eigvalsh(nd_array, check_finite=False)

    # Ordenação por valor absoluto moderna (NumPy >= 1.15)
    # Argsort retorna os índices que ordenariam o array
    idx = np.argsort(np.abs(eigenvalues), axis=-1)

    # Reorganiza os autovalores baseados nos índices
    sorted_eigenvalues = np.take_along_axis(eigenvalues, idx, axis=-1)

    # Move o eixo dos autovalores para o início para desempacotamento fácil: (3, ...)
    sorted_eigenvalues = np.moveaxis(sorted_eigenvalues, -1, 0)

    return sorted_eigenvalues
