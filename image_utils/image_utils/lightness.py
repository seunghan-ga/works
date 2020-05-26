import tensorflow as tf
import numpy as np
import cv2


def lightness_color(img, gamma_1=0.3, gamma_2=1.5):
    # Convert image to RGB to YUV
    img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_YUV)

    # Number of rows and columns
    rows = y.shape[0]
    cols = y.shape[1]

    # Convert image to 0 to 1, then do log(1 + I)
    imgLog = tf.math.log1p(np.array(y, dtype='float') / 255)

    # Create Gaussian mask of sigma = 10
    M = 2 * rows + 1
    N = 2 * cols + 1
    sigma = 10
    (X, Y) = tf.meshgrid(tf.linspace(0., N - 1, N), tf.linspace(0., M - 1, M))
    Xc = tf.math.ceil(N / 2)
    Yc = tf.math.ceil(M / 2)
    gaussianNumerator = (X - Xc) ** 2 + (Y - Yc) ** 2

    # Low pass and high pass filters
    LPF = tf.math.exp(-gaussianNumerator / (2 * sigma * sigma))
    HPF = 1 - LPF

    # Move origin of filters so that it's at the top left corner to match with the input image
    LPF_shift = tf.signal.ifftshift(LPF)
    HPF_shift = tf.signal.ifftshift(HPF)

    # Filter the image and crop
    img_FFT = np.fft.fft2(imgLog, (M, N))
    img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift.numpy(), (M, N)))  # low frequency
    img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift.numpy(), (M, N)))  # high frequency

    # Set scaling factors and add
    gamma1 = gamma_1
    gamma2 = gamma_2
    img_adjusting = gamma1 * img_LF[0:rows, 0:cols] + gamma2 * img_HF[0:rows, 0:cols]

    # Anti-log then rescale to [0,1]
    img_exp = np.expm1(img_adjusting)  # exp(x) + 1
    img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp))
    img_out = np.array(255 * img_exp, dtype='uint8')

    # Convert image to YUV to RGB
    img_YUV[:, :, 0] = img_out
    result = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)

    return result


def lightness_gray(img, gamma_1=0.3, gamma_2=1.5, threshold=65):
    def imclearborder(imgBW, radius):
        # Given a black and white image, first find all of its contours
        imgBWcopy = imgBW.copy()
        image, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Get dimensions of image
        imgRows = imgBW.shape[0]
        imgCols = imgBW.shape[1]

        contourList = []  # ID list of contours that touch the border

        # For each contour...
        for idx in np.arange(len(contours)):
            # Get the i'th contour
            cnt = contours[idx]

            # Look at each point in the contour
            for pt in cnt:
                rowCnt = pt[0][1]
                colCnt = pt[0][0]

                # If this is within the radius of the border
                # this contour goes bye bye!
                check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows - 1 - radius and rowCnt < imgRows)
                check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols - 1 - radius and colCnt < imgCols)

                if check1 or check2:
                    contourList.append(idx)
                    break

        for idx in contourList:
            cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)

        return imgBWcopy

    def bwareaopen(imgBW, areaPixels):
        # Given a black and white image, first find all of its contours
        imgBWcopy = imgBW.copy()
        image, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # For each contour, determine its total occupying area
        for idx in np.arange(len(contours)):
            area = cv2.contourArea(contours[idx])
            if (area >= 0 and area <= areaPixels):
                cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)

        return imgBWcopy

    # Number of rows and columns
    rows = img.shape[0]
    cols = img.shape[1]

    # Remove some columns from the beginning and end
    img = img[:, 59:cols - 20]

    # Number of rows and columns
    rows = img.shape[0]
    cols = img.shape[1]

    # Convert image to 0 to 1, then do log(1 + I)
    imgLog = tf.math.log1p(np.array(img, dtype="float") / 255)

    # Create Gaussian mask of sigma = 10
    M = 2 * rows + 1
    N = 2 * cols + 1
    sigma = 10
    (X, Y) = tf.meshgrid(tf.linspace(0., N - 1, N), tf.linspace(0., M - 1, M))
    centerX = tf.math.ceil(N / 2)
    centerY = tf.math.ceil(M / 2)
    gaussianNumerator = (X - centerX) ** 2 + (Y - centerY) ** 2

    # Low pass and high pass filters
    Hlow = tf.math.exp(-gaussianNumerator / (2 * sigma * sigma))
    Hhigh = 1 - Hlow

    # Move origin of filters so that it's at the top left corner to match with the input image
    # HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    # HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())
    HlowShift = tf.signal.ifftshift(Hlow)
    HhighShift = tf.signal.ifftshift(Hhigh)

    # Filter the image and crop
    If = np.fft.fft2(imgLog, (M, N))
    Ioutlow = np.real(np.fft.ifft2(If.copy() * HlowShift.numpy(), (M, N)))
    Iouthigh = np.real(np.fft.ifft2(If.copy() * HhighShift.numpy(), (M, N)))

    # Set scaling factors and add
    gamma1 = gamma_1
    gamma2 = gamma_2
    Iout = gamma1 * Ioutlow[0:rows, 0:cols] + gamma2 * Iouthigh[0:rows, 0:cols]

    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255 * Ihmf, dtype="uint8")

    # Threshold the image - Anything below intensity 65 gets set to white
    Ithresh = Ihmf2 < threshold
    Ithresh = 255 * Ithresh.astype("uint8")

    # Clear off the border.  Choose a border radius of 5 pixels
    Iclear = imclearborder(Ithresh, 5)

    # Eliminate regions that have areas below 120 pixels
    Iopen = bwareaopen(Iclear, 120)

    return Ihmf2, Ithresh, Iopen


if __name__ == "__main__":
    print("lightness module TEST")
