import numpy as np


def sample_func(im, x, y):
    """
    A sample function that given an image, returns two cropped images based
    on the values of x and y

    Parameters
    ----------
    im: np.ndarray
        An image of size H and W
    x: int
    y: int

    Returns
    -------
    im1: np.ndarray
        A subimage of the given image of size (H - y) and (W - x)
    im2: np.ndarray
        A subimage of the given image of size y and x
    """
    im1 = im[y:, x:, :]
    im2 = im[:y, :x]

    return im1, im2
