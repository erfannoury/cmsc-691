import numpy as np
from skimage.color import rgb2gray
from scipy.signal import convolve2d
from scipy.ndimage import rank_filter
from scipy.stats import norm


def dist2(x, c):
    """
    Calculates squared distance between two sets of points.

    Parameters
    ----------
    x: numpy.ndarray
        Data of shape `(ndata, dimx)`
    c: numpy.ndarray
        Centers of shape `(ncenters, dimc)`

    Returns
    -------
    n2: numpy.ndarray
        Squared distances between each pair of data from x and c, of shape
        `(ndata, ncenters)`
    """
    assert x.shape[1] == c.shape[1], \
        'Data dimension does not match dimension of centers'

    x = np.expand_dims(x, axis=0)  # new shape will be `(1, ndata, dimx)`
    c = np.expand_dims(c, axis=1)  # new shape will be `(ncenters, 1, dimc)`

    # We will now use broadcasting to easily calculate pairwise distances
    n2 = np.sum((x - c) ** 2, axis=-1)

    return n2


def gen_dgauss(sigma):
    """
    Generates the horizontally and vertically differentiated Gaussian filter

    Parameters
    ----------
    sigma: float
        Standard deviation of the Gaussian distribution

    Returns
    -------
    Gx: numpy.ndarray
        First degree derivative of the Gaussian filter across rows
    Gy: numpy.ndarray
        First degree derivative of the Gaussian filter across columns
    """
    f_wid = 4 * np.floor(sigma)
    G = norm.pdf(np.arange(-f_wid, f_wid + 1),
                 loc=0, scale=sigma).reshape(-1, 1)
    G = G.T * G
    Gx, Gy = np.gradient(G)

    Gx = Gx * 2 / np.abs(Gx).sum()
    Gy = Gy * 2 / np.abs(Gy).sum()

    return Gx, Gy


def find_sift(I, circles, enlarge_factor=1.5):
    """
    Compute non-rotation-invariant SITF descriptors of a set of circles

    Parameters
    ----------
    I: numpy.ndarray
        Image
    circles: numpy.ndarray
        An array of shape `(ncircles, 3)` where ncircles is the number of
        circles, and each circle is defined by (x, y, r), where r is the radius
        of the cirlce
    enlarge_factor: float
        Factor which indicates by how much to enlarge the radius of the circle
        before computing the descriptor (a factor of 1.5 or large is usually
        necessary for best performance)

    Returns
    -------
    sift_arr: numpy.ndarray
        Array of SIFT descriptors of shape `(ncircles, 128)`
    """
    I = I.astype(np.float64)
    if I.ndim == 3:
        I = rgb2gray(I)

    NUM_ANGLES = 8
    NUM_BINS = 4
    NUM_SAMPLES = NUM_BINS * NUM_BINS
    ALPHA = 9
    SIGMA_EDGE = 1

    ANGLE_STEP = 2 * np.pi / NUM_ANGLES
    angles = np.arange(0, 2 * np.pi, ANGLE_STEP)

    height, width = I.shape[:2]
    num_pts = circles.shape[0]

    sift_arr = np.zeros((num_pts, NUM_SAMPLES * NUM_ANGLES))

    Gx, Gy = gen_dgauss(SIGMA_EDGE)

    Ix = convolve2d(I, Gx, 'same')
    Iy = convolve2d(I, Gy, 'same')
    I_mag = np.sqrt(Ix ** 2 + Iy ** 2)
    I_theta = np.arctan2(Ix, Iy + 1e-12)

    interval = np.arange(-1 + 1/NUM_BINS, 1 + 1/NUM_BINS, 2/NUM_BINS)
    gridx, gridy = np.meshgrid(interval, interval)
    gridx = gridx.reshape((1, -1))
    gridy = gridy.reshape((1, -1))

    I_orientation = np.zeros((height, width, NUM_ANGLES))

    for i in range(NUM_ANGLES):
        tmp = np.cos(I_theta - angles[i]) ** ALPHA
        tmp = tmp * (tmp > 0)

        I_orientation[:, :, i] = tmp * I_mag

    for i in range(num_pts):
        cx, cy = circles[i, :2]
        r = circles[i, 2]

        gridx_t = gridx * r + cx
        gridy_t = gridy * r + cy
        grid_res = 2.0 / NUM_BINS * r

        x_lo = np.floor(np.max([cx - r - grid_res / 2, 0])).astype(np.int32)
        x_hi = np.ceil(np.min([cx + r + grid_res / 2, width])).astype(np.int32)
        y_lo = np.floor(np.max([cy - r - grid_res / 2, 0])).astype(np.int32)
        y_hi = np.ceil(
            np.min([cy + r + grid_res / 2, height])).astype(np.int32)

        grid_px, grid_py = np.meshgrid(
            np.arange(x_lo, x_hi, 1),
            np.arange(y_lo, y_hi, 1))
        grid_px = grid_px.reshape((-1, 1))
        grid_py = grid_py.reshape((-1, 1))

        dist_px = np.abs(grid_px - gridx_t)
        dist_py = np.abs(grid_py - gridy_t)

        weight_x = dist_px / (grid_res + 1e-12)
        weight_x = (1 - weight_x) * (weight_x <= 1)
        weight_y = dist_py / (grid_res + 1e-12)
        weight_y = (1 - weight_y) * (weight_y <= 1)
        weights = weight_x * weight_y

        curr_sift = np.zeros((NUM_ANGLES, NUM_SAMPLES))
        for j in range(NUM_ANGLES):
            tmp = I_orientation[y_lo:y_hi, x_lo:x_hi, j].reshape((-1, 1))
            curr_sift[j, :] = (tmp * weights).sum(axis=0)
        sift_arr[i, :] = curr_sift.flatten()

    tmp = np.sqrt(np.sum(sift_arr ** 2, axis=-1))
    if np.sum(tmp > 1) > 0:
        sift_arr_norm = sift_arr[tmp > 1, :]
        sift_arr_norm /= tmp[tmp > 1].reshape(-1, 1)

        sift_arr_norm = np.clip(sift_arr_norm, sift_arr_norm.min(), 0.2)

        sift_arr_norm /= np.sqrt(
            np.sum(sift_arr_norm ** 2, axis=-1, keepdims=True))

        sift_arr[tmp > 1, :] = sift_arr_norm

    return sift_arr


def harris(im, sigma, thresh=None, radius=None):
    """
    Harris corner detector

    Parameters
    ----------
    im: numpy.ndarray
        Image to be processed
    sigma: float
        Standard deviation of smoothing Gaussian
    thresh: float (optional)
    radius: float (optional)
        Radius of region considered in non-maximal suppression

    Returns
    -------
    cim: numpy.ndarray
        Binary image marking corners
    r: numpy.ndarray
        Row coordinates of corner points. Returned only if none of `thresh` and
        `radius` are None.
    c: numpy.ndarray
        Column coordinates of corner points. Returned only if none of `thresh`
        and `radius` are None.
    """
    if im.ndim == 3:
        im = rgb2gray(im)

    dx = np.tile([[-1, 0, 1]], [3, 1])
    dy = dx.T

    Ix = convolve2d(im, dx, 'same')
    Iy = convolve2d(im, dy, 'same')

    f_wid = np.round(3 * np.floor(sigma))
    G = norm.pdf(np.arange(-f_wid, f_wid + 1),
                 loc=0, scale=sigma).reshape(-1, 1)
    G = G.T * G
    G /= G.sum()

    Ix2 = convolve2d(Ix ** 2, G, 'same')
    Iy2 = convolve2d(Iy ** 2, G, 'same')
    Ixy = convolve2d(Ix * Iy, G, 'same')

    cim = (Ix2 * Iy2 - Ixy ** 2) / (Ix2 + Iy2 + 1e-12)

    if thresh is None or radius is None:
        return cim
    else:
        size = int(2 * radius + 1)
        mx = rank_filter(cim, -1, size=size)
        cim = (cim == mx) & (cim > thresh)

        r, c = cim.nonzero()

        return cim, r, c


if __name__ == '__main__':
    Gx, Gy = gen_dgauss(3.2)
    print(f'Gx.shape: {Gx.shape}')
    I = np.random.random((480, 640, 3)) * 255
    circles = np.vstack([
        np.random.randint(1, 480, 25),
        np.random.randint(1, 640, 25),
        15 * np.random.random(25)]).T

    sift_arr = find_sift(I, circles)
    print(sift_arr.shape)

    cim, r, c = harris(I, 3.2, thresh=5, radius=3)

    print(f'cim.shape: {cim.shape}')
