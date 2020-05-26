from scipy import stats
import tensorflow as tf
import numpy as np


def resize(image, height, width, method='area', ratio=False, antialias=False):
    """
    Resizes an image to a target width and height.
    :param image: 3D or 4D image
    :param height: height size - integer
    :param width: width size - integer
    :param method: Method to use for resizing image
                   method:"area", "bicubic", "bilinear", "gaussian", "lanczos3", "lanczos5", "mitchellcubic", "nearest"
    :param ratio: Whether to preserve the aspect ratio.
    :param antialias: Whether to use anti-aliasing when resizing.
    :return: Resized and padded image(numpy array).
    """
    resized_tensor = tf.image.resize(image,
                                     [height, width],
                                     method=method,
                                     preserve_aspect_ratio=ratio,
                                     antialias=antialias)
    resized_np = resized_tensor.numpy()

    return resized_np


def resize_with_pad(image, height, width, method='area', antialias=False):
    """
    Resizes and pads an image to a target width and height.
    :param image: 3D or 4D image
    :param height: height size - integer
    :param width: width size - integer
    :param method: Method to use for resizing image
                   method:"area", "bicubic", "bilinear", "gaussian", "lanczos3", "lanczos5", "mitchellcubic", "nearest"
    :param antialias: Whether to use anti-aliasing when resizing.
    :return: Resized and padded image(numpy array).
    """

    resized_tensor = tf.image.resize_with_pad(image, height, width, method=method, antialias=antialias)
    resized_np = resized_tensor.numpy()

    return resized_np


def enhance(x, height, width):
    min = np.min(x)
    max = np.max(x)
    enhance_mat = np.zeros(x.shape)
    for i in range(height):
        for j in range(width):
            if x[i][j] is None:
                pass
            else:
                enhance_mat[i][j] = ((x[i][j] - min) / (max - min))

    return enhance_mat


def remove_outlier(x, method=0):
    remove_mat = x.copy()
    if method == 0:
        z_score = stats.zscore(x)
        remove_mat[remove_mat > z_score] = 0
    elif method == 1:
        q1 = np.percentile(x, 25, interpolation='midpoint')
        q3 = np.percentile(x, 75, interpolation='midpoint')
        iqr = q3 - q1
        remove_mat[remove_mat < (q1 - 1.5 * iqr)] = 0
        remove_mat[remove_mat > (q3 + 1.5 * iqr)] = 0

    return remove_mat


if __name__ == "__main__":
    print("image module TEST")





