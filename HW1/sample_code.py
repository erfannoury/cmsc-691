import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from utils import sample_func


if __name__ == '__main__':
    im1 = Image.open('Yosemite.png')
    im1 = np.asarray(im1)
    print('im1.shape: {} - im1.dtype: {}'.format(im1.shape, im1.dtype))

    im2 = im1.astype(np.float64)
    im3 = im2.astype(np.uint8)

    plt.imshow(im3)
    plt.title('Original image')
    plt.show()

    plt.imshow((im2 + 50).astype(np.uint8))
    plt.title('Brightened image')
    plt.show()

    plt.imshow(np.clip(im2 + 50, a_min=0, a_max=255).astype(np.uint8))
    plt.title('Brightened image with correct clipping')
    plt.show()

    im4, im5 = sample_func(im2, 100, 110)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax1.imshow(im4.astype(np.uint8), aspect='equal')
    ax2.imshow(im5.astype(np.uint8), aspect='equal')
    plt.show()
