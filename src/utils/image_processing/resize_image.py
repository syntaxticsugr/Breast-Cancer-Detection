import numpy as np
from skimage.transform import resize



def resize_image(image, resize_hw):
    """Resize the input image to the specified height and width.

    Parameters:
    ---
        image: ndarray
            Input image as a NumPy array.
        resize_hw: tuple
            Desired height and width as a tuple (height, width).

    Returns:
    ---
        image: ndarray
            Resized image.

    """

    resized_image = resize(image, resize_hw, anti_aliasing=True)
    resized_image = (resized_image*255).astype(np.uint8)

    return resized_image
