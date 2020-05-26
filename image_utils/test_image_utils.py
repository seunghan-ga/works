from __future__ import absolute_import, division, print_function, unicode_literals

from image_utils import image_utils
import tensorflow as tf
import numpy as np
import cv2


def preprocess(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_img.shape
    enhance_mat = image_utils.image.enhance(gray_img, height, width)
    filtered_mat = image_utils.fft.LPF_blur(enhance_mat, 5)

    return filtered_mat


if __name__ == "__main__":
    tf.debugging.set_log_device_placement(True)
    # cv2.setUseOptimized(True)
    # print(cv2.getBuildInformation())

    with tf.device('/GPU:0'):
        stime = cv2.getTickCount()

        # Threshold weight (std)
        thr_k = 1.25
        # Crop padding
        padding = 5

        # Pre processing
        original_img = cv2.imread("samples/sample1.jpg", cv2.IMREAD_UNCHANGED)
        resized_img = image_utils.image.resize(original_img, 256, 256, method='bicubic', antialias=True)
        preprocessed_img = (preprocess(resized_img))[padding:-padding, padding:-padding]
        cv2.imwrite("preprocess_crop.jpg", preprocessed_img * 255)

        # Low Pass Filter (Ref.)
        k = int(np.floor(np.sqrt(np.size(preprocessed_img)) / 6))
        LPF_blur_mat = image_utils.fft.LPF_blur(preprocessed_img, k)
        thr_lpf_mat = image_utils.threshold.std_threshold(LPF_blur_mat, k=thr_k)
        # thr_lpf_mat = image_utils.threshold.otsu_threshold(LPF_blur_mat)

        cv2.imwrite("LPF_blur.jpg", LPF_blur_mat * 255)
        cv2.imwrite("LPF_threshold.jpg", thr_lpf_mat)

        # Discrete Cosine Transform
        reconstruct_mat = image_utils.reconstruction.dct_reconstruction(preprocessed_img, 1)
        thr_dct_mat = image_utils.threshold.std_threshold(reconstruct_mat, k=thr_k)
        # thr_dct_mat = image_utils.threshold.otsu_threshold(reconstruct_mat)

        cv2.imwrite("DCT_bg_reconstruct.jpg", reconstruct_mat * 255)
        cv2.imwrite("DCT_threshold.jpg", thr_dct_mat)

        comp1 = cv2.bitwise_not(cv2.bitwise_xor(thr_dct_mat, thr_lpf_mat))
        comp_1 = np.sum(comp1) / 255

        # Polynomial surface fitting
        polyfit_mat = image_utils.reconstruction.polynomial_reconstruction(preprocessed_img)
        thr_poly_mat = image_utils.threshold.std_threshold(polyfit_mat, k=thr_k)
        # thr_poly_mat = image_utils.threshold.otsu_threshold(polyfit_mat)

        cv2.imwrite("POLY_polyfit.jpg", polyfit_mat * 255)
        cv2.imwrite("POLY_threshold.jpg", thr_poly_mat)

        comp2 = cv2.bitwise_not(cv2.bitwise_xor(thr_poly_mat, thr_lpf_mat))
        comp_2 = np.sum(comp2) / 255

        # Compare background
        print(comp_1, comp_2)
        if comp_1 > comp_2:
            remove_outlier_bg = image_utils.image.remove_outlier(thr_dct_mat, 1)
            final_bg = image_utils.reconstruction.polynomial_reconstruction(thr_dct_mat)
            print("bg1")
        else:
            final_bg = polyfit_mat
            print('bg3')

        etime = cv2.getTickCount()
        print("%s seconds." % ((etime - stime) / cv2.getTickFrequency()))

        cv2.imshow('final background', final_bg)
        cv2.waitKey(0)
