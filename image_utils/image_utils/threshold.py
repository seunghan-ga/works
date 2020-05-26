import numpy as np


def threshold(x, thr=127):
    x[x < thr] = 0
    x[x >= thr] = 255

    return x.astype(np.uint8)


def otsu_threshold(x):
    """
    Otsu's algorithm based binarization.
    :param x: The input array.
    :return:
    """
    cMax, thr = 0, 0
    for t in np.arange(-1, 1, 0.01):
        clsL = x[np.where(x < t)]
        clsH = x[np.where(x >= t)]
        wL = clsL.size / x.size
        wH = clsH.size / x.size
        meanL = np.mean(float(wL))
        meanH = np.mean(float(wH))
        cVal = wL * wH * (meanL - meanH) ** 2
        if cVal > cMax:
            cMax, thr = cVal, t

    thr_mat = x.copy()
    thr_mat[thr_mat < thr] = 0
    thr_mat[thr_mat >= thr] = 255

    return thr_mat.astype(np.uint8)


def std_threshold(x, k=1.0, ttype=2):
    """
    Standard deviation based binarization.
    :param x: The input array.
    :param k: Weight
    :param ttype:
    :return:
    """
    Bd = x.mean() + (k * x.std())
    Dd = x.mean() - (k * x.std())

    thr_mat = x.copy()

    if ttype == 0:
        thr_mat[thr_mat > Bd] = 0
        thr_mat[thr_mat <= Bd] = 255
    elif ttype == 1:
        thr_mat[thr_mat < Dd] = 0
        thr_mat[thr_mat >= Dd] = 255
    elif ttype == 2:
        thr = k * (Bd + Dd) / 2
        thr_mat[thr_mat < thr] = 0
        thr_mat[thr_mat >= thr] = 255

    return thr_mat.astype(np.uint8)


if __name__ == "__main__":
    print("threshold module TEST")
