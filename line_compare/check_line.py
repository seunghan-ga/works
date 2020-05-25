#!/usr/bin/env python3
# coding: utf-8

from pymongo import MongoClient, ASCENDING, DESCENDING
from bson.objectid import ObjectId
from pymongo.errors import BulkWriteError
from datetime import date, datetime, timedelta
import time
import os
import argparse
import cv2
import shutil
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--line_crop_dir', help='cropped data directory.', default='./line_crop_dir')
args = parser.parse_args()

DB_IP = '127.0.0.1'
DB_NAME = 'register'
COLL_NAME = 'reference'


def make_connection(hostip):
    try :
        connection = MongoClient(host=hostip, w=1, tz_aware=False)
        print ("DB Connected : ", hostip)
    except ConnectionFailure as exception:
        print ("Error(%s): Could not connect to MongoDB" % (type(exception).__name__))
        sys.exit(0)
    return connection

def make_insert(db_coll, doc):
    db_coll.insert(doc)

def make_disconnection(conn):
    conn.close()


def check_line(test_file, ref_file):
    test, ref = {}, {}
    with open(test_file, 'r') as f:
        for line in f.readlines():
            name = "{}_{}".format(line.split('_')[0], line.split('_')[1])
            value = float(line.split('_')[-1])
            test[name] = value
    with open(ref_file, 'r') as f:
        for line in f.readlines():
            name = "{}_{}".format(line.split('_')[0], line.split('_')[1])
            value = float(line.split('_')[-1])
            ref[name] = value
    
    res = []
    for key in test.keys():
        d_res = 'none'
        if test[key] > ref[key]:
            d_res = 'thick'
        elif test[key] < ref[key]:
            d_res = 'thin'
        res.append("{} {}".format(key, d_res))
                
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
                            
            for item in res:
                print(item)    
    