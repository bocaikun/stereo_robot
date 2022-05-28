import numpy as np
import matplotlib.pyplot as plt
import os, sys, json, cv2
import glob
import pandas as pd

def normalize(data, in_range, out_range):
    in_range = np.array(in_range).astype(np.float32)
    compressed = (data - in_range.min()) / (in_range.max() - in_range.min())
    stretched = compressed * (out_range[1] - out_range[0]) + out_range[0]
    return stretched

def csv_pack(csv_path, mode=0):
    if mode == 0:
        ypos_range = [0.645, 0.945] #0
        xpos_range = [-0.45, 0.45] #1
        zpos_range = [0.645, 0.665] #2
        rx_range = [1.5707965*0.99, 1.5707965*1.01] #3
        ry_range = [3.141593*0.99, 3.141593*1.01] #4
        rz_range = [1.5707965*0.45, 1.5707965*1.55] #5
        
        csv_pd = pd.read_csv(csv_path, index_col=None, header=None)
        csv_np = csv_pd.to_numpy()

        csv_np[:, 0] = normalize(csv_np[:, 0], ypos_range, [0., 1.])
        csv_np[:, 1] = normalize(csv_np[:, 1], xpos_range, [0., 1.])
        csv_np[:, 2] = normalize(csv_np[:, 2], zpos_range, [0., 1.])
        csv_np[:, 3] = normalize(csv_np[:, 3], rx_range, [0., 1.])
        csv_np[:, 4] = normalize(csv_np[:, 4], ry_range, [0., 1.])
        csv_np[:, 5] = normalize(csv_np[:, 5], rz_range, [0., 1.])
        # print("csv dataset shape", csv_np.shape)
    return csv_np

def img_pack(img_path, size=112, norm=True):
    imgs_list = list(os.listdir(img_path))
    #imgs_num = len(imgs_list)
    imgs_dataset = []
    for i in imgs_list:
        img = cv2.imread(img_path+str(i),cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size, size))
        b,g,r = cv2.split(img)
        img = cv2.merge((r,g,b))
        imgs_dataset.append(img)
    imgs_dataset = np.array(imgs_dataset)
    imgs_dataset = np.transpose(imgs_dataset, [0,3,2,1]) 
    if norm == True:
        imgs_dataset = normalize(imgs_dataset, [0., 255.], [0., 1.])
    # print("img dataset shape:", imgs_dataset.shape)
    return imgs_dataset