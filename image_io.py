import cv2
import numpy as np
import os
from skimage.util import img_as_ubyte, img_as_float

def load_3d_volume(folder:str) -> np.ndarray:
    """Load a TIF image stack as a 3D volume

    Args:
        folder (str): stack directory path name

    Returns:
        np.ndarray: 3D volume with shape (Z,Y,X)
    """
        # Lista todos os arquivos .tif no diretório
    tiff_files = [f for f in os.listdir(folder) if f.endswith('.tif')]
    
    # Ordena os arquivos numericamente
    tiff_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    
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