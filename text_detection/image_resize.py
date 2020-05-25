import cv2
import numpy as np
import os
import shutil


if __name__ == "__main__":


    base_path = 'samples/059A_B/01_etc'
    dst_path = './samples/all/train/01_etc'
    items = os.listdir(base_path)

    for item in items:
        img = cv2.imread(os.path.join(base_path, item))
        print(os.path.join(base_path, item))
        padding = 20
        h, w = img.shape[:2]
        size = h + padding if h > w else w + padding
        matrix = np.float32([[1, 0, (size - w) / 2], [0, 1, (size - h) / 2]])
        dst = cv2.warpAffine(img, matrix, (size, size))
        cv2.imwrite(os.path.join(base_path, "059A_B_{}".format(item)), dst)

