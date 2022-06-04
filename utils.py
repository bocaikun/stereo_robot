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
    ypos_range = [0.645, 1.05] #0
    xpos_range = [-0.45, 0.45] #1
    zpos_range = [0.645, 0.655] #2
    rx_range = [1.5707965*0.99, 1.5707965*1.01] #3
    ry_range = [3.141593*0.99, 3.141593*1.01] #4
    rz_range = [1.5707965*0.45, 1.5707965*1.55] #5
    
    csv_pd = pd.read_csv(csv_path, index_col=None, header=None)
    csv_np = csv_pd.to_numpy()
    if mode == 0:
        csv_np[:, 0] = normalize(csv_np[:, 0], ypos_range, [0., 1.])
        csv_np[:, 1] = normalize(csv_np[:, 1], xpos_range, [0., 1.])
        csv_np[:, 2] = normalize(csv_np[:, 2], zpos_range, [0., 1.])
        csv_np[:, 3] = normalize(csv_np[:, 3], rx_range, [0., 1.])
        csv_np[:, 4] = normalize(csv_np[:, 4], ry_range, [0., 1.])
        csv_np[:, 5] = normalize(csv_np[:, 5], rz_range, [0., 1.])
        # print("csv dataset shape", csv_np.shape)
    if mode == 1:
        csv_np[:, 0] = normalize(csv_np[:, 0], [0., 1.], ypos_range)
        csv_np[:, 1] = normalize(csv_np[:, 1], [0., 1.], xpos_range)
        csv_np[:, 2] = normalize(csv_np[:, 2], [0., 1.], zpos_range)
        csv_np[:, 3] = normalize(csv_np[:, 3], [0., 1.], rx_range)
        csv_np[:, 4] = normalize(csv_np[:, 4], [0., 1.], ry_range)
        csv_np[:, 5] = normalize(csv_np[:, 5], [0., 1.], rz_range)

    return csv_np

def csv_norm(csv_np, mode=0):
    ypos_range = [0.645, 1.05] #0
    xpos_range = [-0.45, 0.45] #1
    zpos_range = [0.645, 0.655] #2
    rx_range = [1.5707965*0.99, 1.5707965*1.01] #3
    ry_range = [3.141593*0.99, 3.141593*1.01] #4
    rz_range = [1.5707965*0.45, 1.5707965*1.55] #5
  
    if mode == 0:
        csv_np[:, 0] = normalize(csv_np[:, 0], ypos_range, [0., 1.])
        csv_np[:, 1] = normalize(csv_np[:, 1], xpos_range, [0., 1.])
        csv_np[:, 2] = normalize(csv_np[:, 2], zpos_range, [0., 1.])
        csv_np[:, 3] = normalize(csv_np[:, 3], rx_range, [0., 1.])
        csv_np[:, 4] = normalize(csv_np[:, 4], ry_range, [0., 1.])
        csv_np[:, 5] = normalize(csv_np[:, 5], rz_range, [0., 1.])
        # print("csv dataset shape", csv_np.shape)
    if mode == 1:
        csv_np[:, 0] = normalize(csv_np[:, 0], [0., 1.], ypos_range)
        csv_np[:, 1] = normalize(csv_np[:, 1], [0., 1.], xpos_range)
        csv_np[:, 2] = normalize(csv_np[:, 2], [0., 1.], zpos_range)
        csv_np[:, 3] = normalize(csv_np[:, 3], [0., 1.], rx_range)
        csv_np[:, 4] = normalize(csv_np[:, 4], [0., 1.], ry_range)
        csv_np[:, 5] = normalize(csv_np[:, 5], [0., 1.], rz_range)

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
    imgs_dataset = np.transpose(imgs_dataset, [0,3,1,2]) 
    if norm == True:
        imgs_dataset = normalize(imgs_dataset, [0., 255.], [0., 1.])
    # print("img dataset shape:", imgs_dataset.shape)
    return imgs_dataset


def img_norm(img, norm=True):
    if norm == True:
        img = normalize(img, [0., 255.], [0., 1.])
    # print("img dataset shape:", imgs_dataset.shape)
    return img

def draw_subplot(
        x, # (time, motor_num)
        y, # (time, motor_num)
        n, 
        yscale = None, 
        linewidth = 1,
        title = None, 
        dashed = None, 
        ylim = [None, None],
        max_iter = None,
        xlabel=None,
        ylabel=None,
        y_legend='',
        y_dashed_legend = '',
        color = None
        ):
    plt.subplot(n[0], n[1], n[2])
    if max_iter is None:
        if x is None:
            plt.plot(y, linewidth=linewidth, label=y_legend, color=color)
        else:
            plt.plot(x, y, linewidth=linewidth, label=y_legend, color=color)
    else:
        if x is None:
            for i in range(max_iter):
                plt.plot(y[:,i], linewidth=linewidth, c=get_colorcode(i), label=y_legend+f'_{i}')
        else:
            for i in range(max_iter):
                plt.plot(x, y[:, i], linewidth=linewidth, c=get_colorcode(i), label=y_legend+f'_{i}')
    
    if dashed is not None:
        if max_iter is None:
            if x is None:
                plt.plot(dashed, linestyle='dashed', linewidth=linewidth, label=y_dashed_legend)
            else:
                plt.plot(x, dashed, linestyle='dashed', linewidth=linewidth, label=y_dashed_legend)
        else:
            if x is None:
                for i in range(max_iter):
                    plt.plot(dashed[:, i], linestyle='dashed', linewidth=linewidth, c=get_colorcode(i), label=y_dashed_legend+f'_{i}')
            else:
                for i in range(max_iter):
                    plt.plot(x, dashed[:, i], linestyle='dashed', linewidth=linewidth, c=get_colorcode(i), label=y_dashed_legend+f'_{i}')

    if title:
        plt.title(title)
    if yscale:
        plt.yscale(yscale)
    plt.ylim(ylim[0], ylim[1])
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if y_legend and y_dashed_legend:
        plt.legend()
    plt.grid()
