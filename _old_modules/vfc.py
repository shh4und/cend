import numpy as np
from scipy import ndimage as ndi
from scipy import linalg
from skimage.util import img_as_ubyte, img_as_float


def create_edge_map(image_3d, mode="sobel"):
    """
    Creates a normalized gradient magnitude map from a 3D image.

    Args:
        image_3d (np.ndarray): The input 3D image volume.
        mode (str, optional): The method for edge detection. Currently, only
                              "sobel" is supported. Defaults to "sobel".

    Returns:
        np.ndarray: A 3D array representing the normalized edge map, with
                    values scaled between 0.0 and 1.0.
    """
    edge_map = np.zeros_like(image_3d, dtype=float)
    image_3d = img_as_float(image_3d)

    if mode == "sobel":
        # Calculate gradients using the Sobel operator for each axis
        sobel_z = ndi.sobel(image_3d, 0)
        sobel_y = ndi.sobel(image_3d, 1)
        sobel_x = ndi.sobel(image_3d, 2)

        # Calculate the gradient magnitude
        edge_map = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)

        # Normalize the edge map to the range [0, 1]
        max_val = edge_map.max()
        min_val = edge_map.min()
        if (max_val - min_val) > 0:
            edge_map = (edge_map - min_val) / (max_val - min_val)
        else:
            edge_map = np.zeros_like(edge_map)

    return img_as_float(edge_map)


def create_vfc_kernel_3d(size, sigma=3.0):
    """
    Creates a 3D Vector Field Convolution (VFC) kernel.

    The kernel is a 3D vector field where each vector points towards the center
    of the kernel. The magnitude of the vectors decreases with distance from
    the center, following a Gaussian falloff.

    Args:
        size (int): The size of the kernel cube (e.g., 5 for a 5x5x5 kernel).
        sigma (float, optional): The standard deviation of the Gaussian
                                 function, controlling the rate of falloff.
                                 Defaults to 3.0.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
        z, y, and x components of the normalized VFC kernel.
    """
    # Create a coordinate grid for the kernel
    half_size = size // 2
    z, y, x = np.mgrid[
        -half_size : half_size + 1,
        -half_size : half_size + 1,
        -half_size : half_size + 1,
    ]

    # Calculate the distance from the center for each point in the grid
    r = np.sqrt(z**2 + y**2 + x**2)
    r[r == 0] = 1e-8  # Avoid division by zero at the center

    # Calculate the Gaussian term
    exp_term = np.exp(-(r**2) / (sigma**2))

    # Calculate the vector components pointing towards the center
    kernel_z = exp_term * (-z / r)
    kernel_y = exp_term * (-y / r)
    kernel_x = exp_term * (-x / r)

    # Normalize the kernel's vector magnitudes to a max of 1
    magnitude = np.sqrt(kernel_z**2 + kernel_y**2 + kernel_x**2)
    max_magnitude = magnitude.max()

    if max_magnitude != 0:
        kernel_z /= max_magnitude
        kernel_y /= max_magnitude
        kernel_x /= max_magnitude

    return kernel_z, kernel_y, kernel_x


def apply_vfc_3d(volume, kernel_z, kernel_y, kernel_x):
    """
    Applies a 3D Vector Field Convolution (VFC) to a volume.

    This function convolves the input volume (typically an edge map) with each
    component of the VFC kernel to produce a vector field.

    Args:
        volume (np.ndarray): The 3D input array (e.g., an edge map).
        kernel_z (np.ndarray): The z-component of the VFC kernel.
        kernel_y (np.ndarray): The y-component of the VFC kernel.
        kernel_x (np.ndarray): The x-component of the VFC kernel.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
        z, y, and x components of the resulting vector field.
    """
    volume = img_as_float(volume)

    # Convolve the volume with each kernel component to get the resulting vector field
    field_z = ndi.convolve(volume, kernel_z, mode="reflect")
    field_y = ndi.convolve(volume, kernel_y, mode="reflect")
    field_x = ndi.convolve(volume, kernel_x, mode="reflect")

    return field_z, field_y, field_x


def medialness_3d(image_3d, kernel_size=4, sigma=2.0):
    """
    Calculates a "medialness" map for a 3D image.

    Medialness is a measure of how likely a voxel is to be on the medial axis
    (skeleton) of a structure. It is computed by applying VFC to an edge map of
    the image. At the medial axis, the resulting VFC vectors converge and cancel
    each other out, leading to a low magnitude. The medialness is therefore
    calculated as (1 - normalized magnitude).

    Args:
        image_3d (np.ndarray): The input 3D image.
        kernel_size (int, optional): The size of the VFC kernel. Defaults to 4.
        sigma (float, optional): The sigma for the VFC kernel's Gaussian.
                                 Defaults to 2.0.

    Returns:
        np.ndarray: A 3D float array where high values indicate a high
                    probability of being on the medial axis.
    """
    edge_map = create_edge_map(image_3d)
    kernel_z, kernel_y, kernel_x = create_vfc_kernel_3d(kernel_size, sigma)
    field_z, field_y, field_x = apply_vfc_3d(edge_map, kernel_z, kernel_y, kernel_x)

    # Calculate the magnitude of the resulting vector field
    magnitude = linalg.norm(np.stack([field_z, field_y, field_x]), axis=0)

    # Normalize the magnitude to the range [0, 1]
    max_val = magnitude.max()
    min_val = magnitude.min()
    if (max_val - min_val) > 0:
        normalized_magnitude = (magnitude - min_val) / (max_val - min_val)
    else:
        normalized_magnitude = np.zeros_like(magnitude)

    # Medialness is high where the vector field magnitude is low
    medial_axis_map = 1.0 - normalized_magnitude
    return img_as_float(medial_axis_map)


def scale_space_medialness_3d(image, kernel_size, scale_space):
    """
    Computes medialness over a range of scales and combines the results.

    This function applies the medialness calculation at multiple sigma values
    (scales) and takes the maximum response at each voxel. This allows for the
    detection of medial axes for structures of varying thicknesses.

    Args:
        image (np.ndarray): The input 3D image.
        kernel_size (int): The size of the VFC kernel to use.
        scale_space (iterable): An iterable of sigma values (scales) to process.

    Returns:
        np.ndarray: The final combined medialness map from all scales.
    """
    max_medialness = np.zeros_like(image, dtype=float)
    for sigma in scale_space:
        current_medialness = medialness_3d(image, kernel_size, sigma)
        max_medialness = np.maximum(max_medialness, current_medialness)
    return img_as_float(max_medialness)


def local_maxima_3d(data, order=1):
    """
    Detects local maxima in a 3D array.
    Source: https://stackoverflow.com/questions/55453110/how-to-find-local-maxima-of-3d-array-in-python

    Args:
        data (np.ndarray): The 3D ndarray to search for local maxima.
        order (int, optional): The number of points on each side of a voxel
                               to consider for the comparison. Defaults to 1.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - coords: An (N, 3) array of the (z, y, x) coordinates of N local maxima.
            - values: An (N,) array of the values at the local maxima.
    """
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0  # Exclude the center pixel

    filtered_data = ndi.maximum_filter(data, footprint=footprint)
    mask_local_maxima = data > filtered_data

    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]

    return coords, values


def create_maxima_image(coords, shape):
    """
    Creates a binary image from a list of coordinates.

    This function is useful for visualizing the locations of detected points,
    such as local maxima.

    Args:
        coords (np.ndarray): An (N, 3) array of (z, y, x) coordinates.
        shape (tuple): The (z, y, x) shape of the output image.

    Returns:
        np.ndarray: A binary 8-bit image where pixels at the given
                    coordinates are set to 255.
    """
    image = np.zeros(shape, dtype=np.uint8)
    # Transpose coordinates to get separate z, y, x index arrays
    indices = tuple(coords.T)
    image[indices] = 255
    return img_as_ubyte(image)
