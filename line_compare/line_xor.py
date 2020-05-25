#!/usr/bin/env python3
# coding: utf-8

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import time
import os
import argparse
import cv2
import shutil
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--line_input_dir', help='read test data directory.', default='./line_input_dir')
parser.add_argument('--line_crop_dir', help='cropped data directory.', default='./line_crop_dir')
args = parser.parse_args()

DB_IP = '127.0.0.1'
DB_NAME = 'register'
COLL_NAME = 'reference'


def make_connection(hostip):
    try:
        connection = MongoClient(host=hostip, w=1, tz_aware=False)
        print("DB Connected : ", hostip)
    except ConnectionFailure as exception:
        print("Error(%s): Could not connect to MongoDB" % (type(exception).__name__))
        sys.exit(0)
    return connection


def make_insert(db_coll, doc):
    db_coll.insert(doc)


def make_disconnection(conn):
    conn.close()


def get_reference_data(file):
    # get db connection
    conn = make_connection(DB_IP)
    db_coll = conn[DB_NAME][COLL_NAME]

    # !! 해당부분은 샴 네트워크 부분 추가 필요.
    doc = {}
    if (file == "test1.png"):
        name_value = "ref.bmp"
    elif (file == "test2.png"):
        name_value = "ref2.bmp"

    doc["name"] = name_value
    cursor = db_coll.find(doc)
    for result_doc in cursor:
        url = result_doc["url"]

    make_disconnection(conn)

    return url


def xor(test_path, ref_url):
    try:
        test = cv2.imread(test_path)
        ref = cv2.imread(ref_url)

        gray_test = cv2.cvtColor(test[500:6585, 5250:14590], cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(ref[500:6585, 5250:14590], cv2.COLOR_BGR2GRAY)

        result_dst = cv2.bitwise_xor(gray_ref, gray_test)

        kernel = np.ones((5, 5), np.uint8)

        _, thresh_result_dst = cv2.threshold(result_dst, 10, 255, cv2.THRESH_BINARY)
        dil_img = cv2.dilate(thresh_result_dst, kernel, iterations=1)

        crop(gray_test, gray_ref, dil_img, test_path.split('/')[-1])

        return 0
    except Exception as e:
        print(e)
        return -1


def crop(gray_test_img, gray_ref_img, dil_img, filename):
    kernel1 = np.ones((2, 2), np.uint8)
    crop_size = 20
    num = 0
    _, contours, hierarchy = cv2.findContours(dil_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    for c, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if len(cnt) > 5:
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            center_x = int((x + x + w) / 2)
            center_y = int((y + y + h) / 2)

            if w > h:
                rec_size = int(w / 2)
                test_crop_img = gray_test_img[center_y - rec_size:center_y + rec_size,
                                center_x - rec_size:center_x + rec_size]
                ref_crop_img = gray_ref_img[center_y - rec_size:center_y + rec_size,
                               center_x - rec_size:center_x + rec_size]
            else:
                rec_size = int(h / 2)
                test_crop_img = gray_test_img[center_y - rec_size:center_y + rec_size,
                                center_x - rec_size:center_x + rec_size]
                ref_crop_img = gray_ref_img[center_y - rec_size:center_y + rec_size,
                               center_x - rec_size:center_x + rec_size]

            _, test_crop_thresh = cv2.threshold(test_crop_img, 20, 255, cv2.THRESH_BINARY)
            _, ref_crop_thresh = cv2.threshold(ref_crop_img, 20, 255, cv2.THRESH_BINARY)

            ###################################################################################################################

            test_dilate = cv2.dilate(test_crop_thresh, kernel1, iterations=1)
            test_erode = cv2.erode(test_dilate, kernel1, iterations=1)

            crop_test_bgr = cv2.cvtColor(test_erode, cv2.COLOR_GRAY2BGR)
            _, test_contours, test_hierarchy = cv2.findContours(test_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
            new_num = 0

            test_crop_path = os.path.join(os.path.abspath(args.line_crop_dir), 'test')
            if not os.path.exists(test_crop_path):
                os.mkdir(test_crop_path)

            with open(test_crop_path + "/{}_test_crop.txt".format(filename), "a") as f:
                for test_c, test_cnt in enumerate(test_contours):
                    rx, ry, rw, rh = cv2.boundingRect(test_cnt)
                    cv2.drawContours(crop_test_bgr, [test_cnt], 0, (0, 0, 255), 1)
                    rec_area = cv2.contourArea(test_cnt)
                    if rec_area == 0.:
                        pass
                    else:
                        # print(filename + "_test_crop" + str(num) + "_" + str(new_num) + "_" + str(rec_area))
                        crop_pos = "{}_{}_{}_{}".format(rx, ry, rw, rh)
                        f.write("crop" + str(num) + "_" + str(new_num) + "_" + crop_pos + "_" + str(rec_area) + "\n")
                        new_num += 1

                cv2.imwrite(test_crop_path + "/{}_crop".format(filename) + str(num) + ".jpg", crop_test_bgr)

            ###################################################################################################################

            ref_dilate = cv2.dilate(ref_crop_thresh, kernel1, iterations=1)
            ref_erode = cv2.erode(ref_dilate, kernel1, iterations=1)

            crop_ref_bgr = cv2.cvtColor(ref_erode, cv2.COLOR_GRAY2BGR)
            _, ref_contours, ref_hierarchy = cv2.findContours(ref_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
            new_num = 0

            ref_crop_path = os.path.join(os.path.abspath(args.line_crop_dir), 'ref')
            if not os.path.exists(ref_crop_path):
                os.mkdir(ref_crop_path)
            with open(ref_crop_path + "/{}_ref_crop.txt".format(filename), "a") as f:
                for ref_c, ref_cnt in enumerate(ref_contours):
                    rx, ry, rw, rh = cv2.boundingRect(ref_cnt)
                    cv2.drawContours(crop_test_bgr, [ref_cnt], 0, (0, 0, 255), 1)
                    rec_area = cv2.contourArea(ref_cnt)
                    if rec_area == 0.:
                        pass
                    else:
                        # print(filename + "_ref_crop" + str(num) + "_" + str(new_num) + "_" + str(rec_area))
                        crop_pos = "{}_{}_{}_{}".format(rx, ry, rw, rh)
                        f.write("crop" + str(num) + "_" + str(new_num) + "_" + crop_pos + "_" + str(rec_area) + "\n")
                        new_num += 1

                cv2.imwrite(ref_crop_path + "/{}_crop".format(filename) + str(num) + ".jpg", crop_test_bgr)
                num += 1


if __name__ == "__main__":
    abs_path = os.path.abspath(args.line_input_dir)
    file_list = os.listdir(abs_path)

    crop_path = os.path.abspath(args.line_crop_dir)
    if os.path.exists(crop_path):
        shutil.rmtree(crop_path)
    os.mkdir(crop_path)

    for file in file_list:
        url = get_reference_data(file)
        test = os.path.join(abs_path, file)

        print("url is ...", url)
        print(test)

        res = xor(test, url)
        print(res)
