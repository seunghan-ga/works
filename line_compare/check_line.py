#!/usr/bin/env python3
# coding: utf-8

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--line_input_dir', help='read test data directory.', default='./line_input_dir')
parser.add_argument('--line_crop_dir', help='cropped data directory.', default='./line_crop_dir')
args = parser.parse_args()

DB_IP = '127.0.0.1'
DB_NAME = 'defect'
COLL_NAME = 'result'


def make_connection(hostip):
    try:
        connection = MongoClient(host=hostip, w=1, tz_aware=False)
        print("DB Connected : ", hostip)
    except ConnectionFailure as exception:
        print("Error(%s): Could not connect to MongoDB" % (type(exception).__name__))
        sys.exit(0)
    return connection


def make_insert(db_coll, doc):
    db_coll.insert_one(doc)


def make_disconnection(conn):
    conn.close()


def put_defect_data(data):
    try:
        conn = make_connection(DB_IP)
        db_coll = conn[DB_NAME][COLL_NAME]
        make_insert(db_coll, data)
        make_disconnection(conn)
    except Exception as e:
        print(e)


def check_line(test_file, ref_file):
    test, ref = {}, {}
    with open(test_file, 'r') as f:
        for line in f.readlines():
            name = "{}_{}".format(line.split('_')[0], line.split('_')[1])
            position = "{} {} {} {}".format(line.split('_')[2], line.split('_')[3],
                                            line.split('_')[4], line.split('_')[5])
            value = float(line.split('_')[-1])
            test[name] = [value, position]
    with open(ref_file, 'r') as f:
        for line in f.readlines():
            name = "{}_{}".format(line.split('_')[0], line.split('_')[1])
            position = "{} {} {} {}".format(line.split('_')[2], line.split('_')[3],
                                            line.split('_')[4], line.split('_')[5])
            value = float(line.split('_')[-1])
            ref[name] = [value, position]

    res = []
    for key in test.keys():
        d_res = 'none'
        if test[key] > ref[key]:
            d_res = 'thick'
        elif test[key] < ref[key]:
            d_res = 'thin'
        res.append([test[key][1], d_res])

    return res


if __name__ == "__main__":
    test_dir = os.path.join(os.path.abspath(args.line_crop_dir), 'test')
    ref_dir = os.path.join(os.path.abspath(args.line_crop_dir), 'ref')

    test_crop_files = [item for item in os.listdir(test_dir) if '.txt' in item]
    ref_crop_files = [item for item in os.listdir(ref_dir) if '.txt' in item]

    sorted_test_crops = sorted(test_crop_files)
    sorted_ref_crops = sorted(ref_crop_files)
    size = len(sorted_test_crops)

    for idx in range(size):
        if sorted_test_crops[idx].split('_')[0] == sorted_ref_crops[idx].split('_')[0]:
            res = check_line(os.path.join(test_dir, sorted_test_crops[idx]),
                             os.path.join(ref_dir, sorted_ref_crops[idx]))

            thin_defect, thick_defect = [], []
            for item in res:
                if item[1] == 'thin':
                    pos_str = item[0].split(' ')
                    position = {'x': pos_str[0], 'y': pos_str[1],
                                'w': pos_str[2], 'h': pos_str[3]}
                    thin_defect.append(position)
                elif item[1] == 'thick':
                    pos_str = item[0].split(' ')
                    position = {'x': pos_str[0], 'y': pos_str[1],
                                'w': pos_str[2], 'h': pos_str[3]}
                    thick_defect.append(position)

            insert_data = {}
            insert_data['name'] = sorted_test_crops[idx].split('_')[0]
            insert_data['url'] = os.path.join(args.line_input_dir, insert_data['name'])
            insert_data['thin_defect'] = thin_defect
            insert_data['thick_defect'] = thick_defect

            put_defect_data(insert_data)
