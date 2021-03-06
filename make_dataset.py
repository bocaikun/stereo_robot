import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import os, sys, json, cv2
import pandas as pd
from utils import img_pack, csv_pack

def creat_dir(data_id):
    if not os.path.exists('D:/data'):
        os.makedirs('D:/data')
    if not os.path.exists('D:/data/normalize'):
        os.makedirs('D:/data/normalize/')
    if not os.path.exists('D:/data/normalize/%s'%data_id):
        os.makedirs('D:/data/normalize/%s'%data_id)
    if not os.path.exists('D:/data/normalize/%s/train'%data_id):
        os.makedirs('D:/data/normalize/%s/train'%data_id)
    if not os.path.exists('D:/data/normalize/%s/test'%data_id):
        os.makedirs('D:/data/normalize/%s/test'%data_id)

def make_csv(path_list, mode=0):
    csv_list = []
    for path in path_list:
        csv = csv_pack(path)
        csv_list.append(csv)
    return csv_list

def make_img(path_list,size=112, norm=True):
    img_list = []
    for path in path_list:
        img = img_pack(path)
        img_list.append(img)
    return img_list
    

def main(data_id):
    train_csv_path_list = glob.glob('D:/data/original/{}/train/*/csv/*.csv'.format(data_id))
    train_left_path_list = glob.glob('D:/data/original/{}/train/*/image/left_images/'.format(data_id))
    train_right_path_list = glob.glob('D:/data/original/{}/train/*/image/right_images/'.format(data_id))
    test_csv_path_list = glob.glob('D:/data/original/{}/test/*/csv/*.csv'.format(data_id))
    test_left_path_list = glob.glob('D:/data/original/{}/test/*/image/left_images/'.format(data_id))
    test_right_path_list = glob.glob('D:/data/original/{}/test/*/image/right_images/'.format(data_id))

    creat_dir(data_id)

    train_csv_list = make_csv(train_csv_path_list)
    print("train csv done")
    train_left_list = make_img(train_left_path_list)
    print("train left done")
    train_right_list = make_img(train_right_path_list)
    print("train right done")
    test_csv_list = make_csv(test_csv_path_list)
    print("test csv done")
    test_left_list = make_img(test_left_path_list)
    print("test left done")
    test_right_list = make_img(test_right_path_list)
    print("test right done")

    print(np.array(test_csv_list).shape, np.array(test_left_list).shape)

    np.save('D:/data/normalize/{}/train/left_img.npy'.format(data_id),np.array(train_left_list))
    np.save('D:/data/normalize/{}/train/right_img.npy'.format(data_id),np.array(train_right_list))
    np.save('D:/data/normalize/{}/train/joint.npy'.format(data_id), np.array(train_csv_list))
    np.save('D:/data/normalize/{}/test/left_img.npy'.format(data_id),np.array(test_left_list))
    np.save('D:/data/normalize/{}/test/right_img.npy'.format(data_id),np.array(test_right_list))
    np.save('D:/data/normalize/{}/test/joint.npy'.format(data_id), np.array(test_csv_list))

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_id')
    args = parser.parse_args()

    main(args.data_id)