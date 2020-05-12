import os
import cv2
import numpy as np


def silce(img, rows, cols, prefix):
    h, w, _ = img.shape if len(img.shape) > 3 else np.expand_dims(img, 2).shape
    st_h = [np.round(i * (h / rows)) for i in range(rows)]
    ed_h = [np.round(i + (h / rows)) for i in st_h]
    st_w = [np.round(i * (w / cols)) for i in range(cols)]
    ed_w = [np.round(i + (w / cols)) for i in st_w]

    for i in range(len(st_h)):
        for j in range(len(st_w)):
            tmp = img[int(st_h[i]):int(ed_h[i]), int(st_w[j]):int(ed_w[j])]
            filename = "{}_{}-{}.png".format(prefix, i, j)
            cv2.imwrite(os.path.abspath(os.path.join("./data/slice", filename)), tmp)


def flip(img, mode=0):
    # mode: 0 up/down, 1 left/right
    assert len(img.shape) == 3, "not 3 dimension"
    assert img.shape[2] == 3, "not RGB"
    to_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = cv2.flip(to_rgb, mode)

    return res


if __name__ == "__main__":
    src = "../utils/data/original"
    file_list = os.listdir(src)
    for item in file_list:
        img = cv2.imread(os.path.abspath(os.path.join(src, item)))
        # img_flip = flip(img, mode=1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thr, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        prefix = item.split('.')[0]
        silce(binary, 2, 3, prefix)

