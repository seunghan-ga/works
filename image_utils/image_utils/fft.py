from scipy.fftpack import dct, idct
import numpy as np
import cv2


def dct2d(x):
    """
    Return the Discrete Cosine Transform of arbitrary type sequence x.
    :param x: The input array.
    :return: The transformed input array.
    """
    return dct(dct(x.T, norm='ortho').T, norm='ortho').astype(np.float)


def idct2d(x):
    """
    Return the Inverse Discrete Cosine Transform of an arbitrary type sequence.
    :param x: The input array.
    :return: The transformed input array.
    """
    return idct(idct(x.T, norm='ortho').T, norm='ortho').astype(np.float)


def LPF(x, height, width, cutoff):
    """
     A filter that passes signals with a frequency lower than a selected cutoff frequency.
    :param x: The input array.
    :param height: A height in input array.
    :param width: A width in input array.
    :param cutoff: Cutoff frequency.
    :return:
    """
    ret = x.copy()
    for i in range(width):
        for j in range(height):
            if np.square(i) + np.square(j) > np.square(cutoff):
                ret[i][j] = 1 / (1 + np.square((np.square(i) + np.square(j)) / cutoff))

    return ret


def LPF_blur(x, k=5):
    """
    A filter that averages out rapid changes in intensity.
    :param x: The input array.
    :param k: Kernel size.
    :return: The filtered input array.
    """
    ksize = (k * 2) + 1
    kernel = np.ones((ksize, ksize), np.float32) / np.square(ksize)
    ret = cv2.filter2D(x, -1, kernel)

    return ret


if __name__ == "__main__":
    print("fft module TEST")
