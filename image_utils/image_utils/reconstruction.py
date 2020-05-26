import numpy as np


def dct_reconstruction(x, cutoff=1.0):
    """
    Discrete cosine transform based background reconstruction.
    :param x: The input array.
    :param cutoff: Cutoff frequency.
    :return: Reconstructed background array.
    """
    height, width = x.shape
    dct_mat = IDA.image_utils.fft.dct2d(x)
    cutoff_mat = IDA.image_utils.fft.LPF(dct_mat, height, width, cutoff)
    idct_mat = IDA.image_utils.fft.idct2d(cutoff_mat)

    return idct_mat


def polynomial_reconstruction(x):
    """

    :param x: The input array.
    :return: Polynomial surface fitted array.
    """
    polyfit_mat = np.zeros(x.shape)
    for i in range(x.shape[0]):
        coefs = np.polyfit(range(x.shape[0]), x[i].T, 8)
        p = np.poly1d(coefs)
        for j in range(x.shape[1]):
            polyfit_mat[i][j] = np.polyval(p, j)

    # fits = np.polynomial.polynomial.polyfit(range(img.shape[0]), img, 8)
    # polyfit_mat = np.polynomial.polynomial.polyval2d(img, img, fits)

    return polyfit_mat


if __name__ == "__main__":
    print("reconstruction module TEST")
