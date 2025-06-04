import numpy as np
from scipy import ndimage as ndi
from scipy import linalg
from skimage.util import img_as_ubyte, img_as_float


def create_edge_map(image3d, mode="sobel"):
    edge_map = np.zeros_like(image3d, dtype=float)
    image3d = img_as_float(image3d)
    if mode == "sobel":
        # Calcular gradientes usando Sobel
        sobelz = ndi.sobel(image3d, 0)
        sobely = ndi.sobel(image3d, 1)
        sobelx = ndi.sobel(image3d, 2)

        # Magnitude do gradiente
        edge_map = np.sqrt(sobelx**2 + sobely**2 + sobelz**2)
        if (edge_map.max() - edge_map.min()) != 0:
            # Normalizar para [0,1]
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())
        else:
            edge_map = np.zeros_like(edge_map)

    return img_as_float(edge_map)


def create_vfc_kernel_3d(size, sigma=3.0):
    # Criar grid de coordenadas
    z, y, x = np.mgrid[
        -size // 2 : np.ceil(size // 2) + 1,
        -size // 2 : np.ceil(size // 2) + 1,
        -size // 2 : np.ceil(size // 2) + 1,
    ]

    # Calcular r
    r = np.sqrt(z**2 + y**2 + x**2)
    r[r == 0] = 1e-8  # Evitar divisão por zero

    # Calcular termo exponencial
    exp_term = np.exp(-(r**2) / (sigma**2))

    # Calcular componentes do vetor
    kz = exp_term * (-z / r)
    ky = exp_term * (-y / r)
    kx = exp_term * (-x / r)

    # Normalizar
    magnitude = np.sqrt(kz**2 + ky**2 + kx**2)
    max_mag = magnitude.max()

    if max_mag != 0:
        kz = kz / max_mag
        ky = ky / max_mag
        kx = kx / max_mag

    return kz, ky, kx


def apply_vfc_3d(volume, kz, ky, kx):
    """
    Improved parallel VFC computation with chunking

    Args:
        volume: 3D input array (z,y,x)
        ky, kx: VFC kernels
        num_processes: Number of processes to use
        chunk_size: Number of slices per chunk
    """
    volume = img_as_float(volume)
    # Initialize output arrays
    fz = ndi.convolve(volume, kx, mode="reflect")
    fy = ndi.convolve(volume, ky, mode="reflect")
    fx = ndi.convolve(volume, kz, mode="reflect")

    return fz, fy, fx


def medialness_3d(
    image3d,
    kernel_size=4,
    sigma=2.0,
):
    edge_map = create_edge_map(image3d)
    kz, ky, kx = create_vfc_kernel_3d(kernel_size, sigma)
    fz, fy, fx = apply_vfc_3d(edge_map, kz, ky, kx)

    # Calculate vector field magnitude
    magnitude = linalg.norm(np.stack([fz, fy, fx]), axis=0)

    # Normalize to [0,1]
    if (magnitude.max() - magnitude.min()) != 0:
        normalize_magnitude = (magnitude - magnitude.min()) / (
            magnitude.max() - magnitude.min()
        )
    else:
        normalize_magnitude = np.zeros_like(magnitude)

    medial_axis = 1 - normalize_magnitude
    return img_as_float(medial_axis)


def scale_space_medialness_3d(image, kernel_size, scale_space):
    medialness_result = np.zeros_like(image, dtype=float)
    for sigma in scale_space:
        current_medialness = medialness_3d(image, kernel_size, sigma)
        # medialness_result += current_medialness
        medialness_result = np.maximum(medialness_result, current_medialness)
    # medialness_result /= np.sqrt(scale_space**2).sum()
    return img_as_float(medialness_result)


# From: https://stackoverflow.com/questions/55453110/how-to-find-local-maxima-of-3d-array-in-python
def local_maxima_3D(data, order=1):
    """Detects local maxima in a 3D array

    Parameters
    ---------
    data : 3d ndarray
    order : int
        How many points on each side to use for the comparison

    Returns
    -------
    coords : ndarray
        coordinates of the local maxima
    values : ndarray
        values of the local maxima
    """
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    # footprint=ndimage.generate_binary_structure(size, 1)
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint)
    mask_local_maxima = data > filtered

    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]

    return coords, values


def create_maxima_image(coords, shape):
    """
    Creates binary image from coordinates

    Args:
        coords: Nx3 array of coordinates
        shape: Shape of output image (z,y,x)
    """
    img = np.zeros(shape, dtype=np.uint8)
    # Convert coordinates to tuple of index arrays
    indices = tuple(coords.T)  # Transpose to get separate x,y,z arrays
    img[indices] = 255
    return img_as_ubyte(img)
