import cv2
import numpy as np
import os
from skimage.util import img_as_ubyte
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format=" - %(message)s")


def load_3d_volume(folder_path: str) -> np.ndarray:
    """
    Loads a sequence of TIFF files from a directory into a 3D NumPy array.

    The files are sorted numerically based on the digits in their filenames
    to ensure the correct stacking order.

    Args:
        folder_path (str): The path to the directory containing the TIFF stack.

    Returns:
        np.ndarray: A 3D volume with shape (Z, Y, X), where Z is the number
                    of images in the stack.

    Raises:
        ValueError: If any image file cannot be read or if the folder is empty.
    """
    # List all .tif files in the directory
    try:
        tiff_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".tif")]
        if not tiff_files:
            raise ValueError(f"No .tif files found in directory: {folder_path}")

        # Sort files numerically based on digits in the filename
        tiff_files.sort(key=lambda x: int("".join(filter(str.isdigit, x))))

    except OSError as e:
        raise ValueError(f"Cannot access directory {folder_path}: {e}")

    # Load the first image to determine dimensions
    first_image_path = os.path.join(folder_path, tiff_files[0])
    first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    if first_image is None:
        raise ValueError(f"Could not read the first image: {tiff_files[0]}")

    # Initialize the 3D volume (stack)
    height, width = first_image.shape
    depth = len(tiff_files)
    image_stack = np.zeros((depth, height, width), dtype=np.uint8)
    image_stack[0] = first_image

    # Load the remaining images into the stack
    for z_index, filename in enumerate(tiff_files[1:], 1):
        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {filename}")
        image_stack[z_index] = img

    return image_stack


def save_3d_volume(image_stack: np.ndarray, save_path: str, prefix: str = "") -> bool:
    """
    Saves a 3D image stack as a sequence of individual TIFF files.

    Args:
        image_stack (np.ndarray): The 3D image volume with shape (Z, Y, X).
        save_path (str): The directory path where the images will be saved.
        prefix (str, optional): A prefix to add to each filename. Defaults to "".

    Returns:
        bool: True if all images were saved successfully, False otherwise.
    """
    try:
        os.makedirs(save_path, exist_ok=True)
        image_stack = img_as_ubyte(image_stack)

        # Calculate zero-padding based on the number of images for correct sorting
        num_digits = len(str(len(image_stack)))

        for idx, image_slice in enumerate(image_stack):
            # Filenames are 1-indexed for convenience
            filename = f"{prefix}{str(idx + 1).zfill(num_digits)}.tif"
            filepath = os.path.join(save_path, filename)

            if not cv2.imwrite(filepath, image_slice):
                print(f"Error: Failed to save image slice at index {idx} to {filepath}")
                return False

        print(f"Successfully saved {len(image_stack)} images to {save_path}")
        return True

    except Exception as e:
        print(f"An error occurred while saving the image stack: {e}")
        return False


def show_stack_basic(image_stack: np.ndarray):
    """
    Displays each slice of a 3D image stack in a separate window.

    Args:
        image_stack (np.ndarray): A 3D NumPy array (Z, Y, X) or a single 2D image.
    """
    if len(image_stack) == 2:
        # If a single 2D image is passed, wrap it in a list to make it iterable
        image_stack = [image_stack]

    for idx, image_slice in enumerate(image_stack):
        window_name = f"Image Slice {idx}"
        cv2.imshow(window_name, image_slice)

    logging.info("Press any key to close all image windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_stack_interactive(image_stack: np.ndarray, show_multiple=False):
    """
    Provides an interactive slide-show viewer for a 3D image stack.

    Keyboard Controls:
        - 'd': Move to the next image in the stack.
        - 'a': Move to the previous image in the stack.
        - 'q': Quit the viewer.

    Args:
        image_stack (np.ndarray): The 3D image stack to display.
        show_multiple (bool): If True, displays the previous, current, and
                              next slices simultaneously. Defaults to False.
    """
    index = 0
    num_images = len(image_stack)
    logging.info("Viewing slices. Use 'a' or 'd' to navigate, 'q' to quit.")
    while True:
        if show_multiple:
            # Display three windows: previous, current, and next
            prev_index = (index - 1 + num_images) % num_images
            next_index = (index + 1) % num_images

            cv2.imshow(f"Slice {prev_index} (Previous)", image_stack[prev_index])
            cv2.imshow(f"Slice {index} (Current)", image_stack[index])
            cv2.imshow(f"Slice {next_index} (Next)", image_stack[next_index])
        else:
            # Display a single window for the current image
            cv2.imshow(f"Slice {index}/{num_images - 1}", image_stack[index])

        key = cv2.waitKey(0)

        # Close current windows before opening new ones
        cv2.destroyAllWindows()

        if key == ord("q"):
            break
        elif key == ord("d"):  # Move to the next image
            index = (index + 1) % num_images
        elif key == ord("a"):  # Move to the previous image
            index = (index - 1 + num_images) % num_images

    cv2.destroyAllWindows()
