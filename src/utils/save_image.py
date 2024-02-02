import numpy as np
from skimage import io
from utils.create_directory import create_directory



def save_image(image, name, dir):
    """Save the input image with a given name to the specified directory.

    Parameters:
    ---
        image: ndarray
            Input image as a NumPy array.
        name: string
            Desired name for the saved image.
        dir: string
            Directory path where the image will be saved.

    Returns:
    ---
        None

    """

    create_directory(dir)

    io.imsave(f"{dir}/{name}", (image*255).astype(np.uint8))
