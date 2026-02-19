import numpy as np
from scipy import ndimage as ndi


def compute_hessian_matrix(nd_array, sigma=1, scale=False):
    """
    Computa a matriz Hessiana usando derivadas Gaussianas diretas.
    Melhora performance e precisão numérica em relação a np.gradient.
    """
    ndim = nd_array.ndim
    if ndim != 3:
        raise ValueError("Apenas arrays 3D são suportados nesta implementação otimizada.")

    # A matriz Hessiana em 3D tem 6 componentes únicos: xx, yy, zz, xy, xz, yz
    # Ordem das derivadas para scipy.ndimage.gaussian_filter
    orders = [
        (2, 0, 0),  # xx
        (0, 2, 0),  # yy
        (0, 0, 2),  # zz
        (1, 1, 0),  # xy
        (1, 0, 1),  # xz
        (0, 1, 1),  # yz
    ]
    scale = False
    # Fator de normalização de escala (Lindeberg)
    scale_factor = (sigma**2) if scale else 1.0

    # Dicionário para armazenar as derivadas calculadas
    H_elems = {}

    for order in orders:
        # Calcula a derivada diretamente via convolução gaussiana
        der = ndi.gaussian_filter(nd_array, sigma=sigma, order=order)
        if scale:
            der *= scale_factor
        H_elems[order] = der

    # Aloca a matriz Hessiana: shape (..., 3, 3)
    # Movemos os eixos 3x3 para o final para compatibilidade com np.linalg.eigvalsh
    hessian_shape = nd_array.shape + (ndim, ndim)
    hessian = np.zeros(hessian_shape, dtype=nd_array.dtype)

    # Preenche a matriz simétrica
    # Mapeamento de (x,y,z) para índices (0,1,2)
    # Hxx
    hessian[..., 0, 0] = H_elems[(2, 0, 0)]
    # Hyy
    hessian[..., 1, 1] = H_elems[(0, 2, 0)]
    # Hzz
    hessian[..., 2, 2] = H_elems[(0, 0, 2)]
    # Hxy e Hyx
    hessian[..., 0, 1] = hessian[..., 1, 0] = H_elems[(1, 1, 0)]
    # Hxz e Hzx
    hessian[..., 0, 2] = hessian[..., 2, 0] = H_elems[(1, 0, 1)]
    # Hyz e Hzy
    hessian[..., 1, 2] = hessian[..., 2, 1] = H_elems[(0, 1, 1)]

    return hessian


def absolute_hessian_eigenvalues(nd_array, sigma=1, scale=False):
    from .utils import absolute_eigenvaluesh

    hessian = compute_hessian_matrix(nd_array, sigma=sigma, scale=scale)
    return absolute_eigenvaluesh(hessian)
