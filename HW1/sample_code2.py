import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


if __name__ == '__main__':
    mu = 0
    sigma = 3
    x = np.arange(start=-3 * sigma, stop=3 * sigma + 1, step=1,
                  dtype=np.float32)
    y = np.exp(- ((x - mu) ** 2) / (2 * sigma ** 2))
    y /= y.sum()

    plt.plot(x, y)
    plt.title('X vs. Y')
    plt.show()

    y = np.expand_dims(y, 1)  # To change the shape from (19, ) to (19, 1)
    # Reshape could be used, too:
    #   (y = y.reshape((-1, 1))).
    # (-1) tells numpy to calculate the correct value for that axis
    # automatically such that the total number of elements in the matrix
    # remains unchanged

    y2 = y @ y.T  # The `@` does matrix multiplication, you can also use
    # np.matmul(y, y.T)

    plt.imshow(y2.astype(np.uint8))
    plt.title('2D filter')
    plt.show()

    y2_norm = (y2 - y2.min()) / (y2.max() - y2.min()) * 255
    plt.imshow(y2_norm.astype(np.uint8))
    plt.title('2D filter (color scale normalized)')
    plt.show()

    im = np.asarray(Image.open('Moire_small.jpg'))

    plt.imshow(im)
    plt.title('Original image')
    plt.show()

    # Create a brighter image
    im2 = np.zeros_like(im, dtype=np.float32)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im2[i, j, :] = im[i, j, :] + 100.0  # I have used 100.0 (a float)
    # instead of 100 so that upcasting happens before adding the values. If 100
    # was used the addition was done between two number of type uint8 and
    # therefore overflow would happen. However, by using 100.0 we force pixel
    # values from im to be first upcasted to float32 and then added to 100.0
    print(im2.min(), im2.max())

    # Faster way to do it:
    # im2 = im + 100

    plt.imshow(np.clip(im2, 0, 255).astype(np.uint8))
    plt.title('Brightened image')
    plt.show()

    # Create a shifted iamge
    im3 = np.zeros_like(im)
    for i in range(10, im.shape[0]):
        for j in range(im.shape[1]):
            im3[i - 10, j, :] = im[i, j, :]

    # Faster way to do it:
    # im3 = np.zeros_like(im)
    # im3[10:, :, :] = im[:-10, :, :]

    plt.imshow(im3)
    plt.title('Shifted image')
    plt.show()
