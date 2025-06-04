import cv2
import numpy as np
import os
from skimage.util import img_as_ubyte, img_as_float


def load_3d_volume(folder: str) -> np.ndarray:
    """Load a TIF image stack as a 3D volume

    Args:
        folder (str): stack directory path name

    Returns:
        np.ndarray: 3D volume with shape (Z,Y,X)
    """
    # Lista todos os arquivos .tif no diretório
    tiff_files = [f for f in os.listdir(folder) if f.endswith(".tif")]

    # Ordena os arquivos numericamente
    tiff_files.sort(key=lambda x: int("".join(filter(str.isdigit, x))))

    # Carrega primeira imagem para obter dimensões
    first_img = cv2.imread(os.path.join(folder, tiff_files[0]), cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        raise ValueError(f"Não foi possível ler a imagem {tiff_files[0]}")

    # Inicializa array 3D
    height, width = first_img.shape
    depth = len(tiff_files)
    stack = np.zeros((depth, height, width), dtype=np.uint8)
    stack[0] = first_img

    # Carrega as demais imagens
    for z, filename in enumerate(tiff_files[1:], 1):
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Não foi possível ler a imagem {filename}")
        stack[z] = img

    return stack


def download(images: np.ndarray, path: str, prefix: str = "") -> bool:
    """Downloads an image stack as individual TIFF files.

    Args:
        images (np.ndarray): 3D image stack with shape (Z,Y,X)
        path (str): Directory path for saving images
        prefix (str, optional): Prefix for filenames. Defaults to ""

    Returns:
        bool: True if successful, False if any error occurs
    """
    try:
        os.makedirs(path, exist_ok=True)

        # Add zero padding to maintain correct order
        n_digits = len(str(len(images)))

        for idx, img in enumerate(images):
            filename = f"{prefix}{str(idx+1).zfill(n_digits)}.tif"
            filepath = os.path.join(path, filename)

            if not cv2.imwrite(filepath, img):
                print(f"Failed to save image at index {idx}")
                return False
        print(f"Downloaded {len(images)} items at {path}")
        return True

    except Exception as e:
        print(f"Error saving stack: {str(e)}")
        return False


def simple_imshow(imgs):
    if len(np.array(imgs).shape) < 3:
        stack3d = np.array([imgs])
    else:
        stack3d = imgs
    x = 0
    for img in stack3d:
        cv2.imshow(f"img 0{x}", img)
        x += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return True


def slide_imshow(image_stack, multiple_windows=False):
    index = 0
    num_images = len(image_stack)

    if not multiple_windows:
        while True:
            curr_window_name = f"image[{index}] | press X to quit"

            cv2.imshow(curr_window_name, image_stack[index])
            key = cv2.waitKey(0)

            if key == ord("x"):  # Press 'x' to quit
                break
            elif key == ord("e"):  # Press 'n' for next image
                cv2.destroyWindow(curr_window_name)
                index = (index + 1) % num_images
            elif key == ord("q"):  # Press 'p' for previous image
                cv2.destroyWindow(curr_window_name)
                index = (index - 1) % num_images
    else:
        while True:
            prev_index = (index - 1) % num_images
            next_index = (index + 1) % num_images

            prev_window_name = f"image[{prev_index}] | press X to quit"
            curr_window_name = f"image[{index}] | press X to quit"
            next_window_name = f"image[{next_index}] | press X to quit"
            cv2.imshow(prev_window_name, image_stack[prev_index])
            cv2.imshow(curr_window_name, image_stack[index])
            cv2.imshow(next_window_name, image_stack[next_index])
            key = cv2.waitKey(0)

            if key == ord("x"):  # Press 'x' to quit
                break
            elif key == ord("e"):  # Press 'n' for next image
                cv2.destroyWindow(prev_window_name)
                cv2.destroyWindow(curr_window_name)
                cv2.destroyWindow(next_window_name)
                index = (index + 1) % num_images
            elif key == ord("q"):  # Press 'p' for previous image
                cv2.destroyWindow(prev_window_name)
                cv2.destroyWindow(curr_window_name)
                cv2.destroyWindow(next_window_name)
                index = (index - 1) % num_images
    cv2.destroyAllWindows()
