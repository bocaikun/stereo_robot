import numpy as np
import matplotlib.pyplot as plt
import os, sys, json, cv2, datetime
import torch
import torch.nn as nn
import pandas as pd
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import argparse

from model.stereo_hiruma import stereo_hiruma_att

class dataset(Dataset):
    def __init__(self,left, right, pos):
        self.left = left
        self.right = right
        self.pos = pos
        assert self.left.shape[0] == self.pos.shape[0] == self.right.shape[0]

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        return self.left[idx],self.right[idx],self.pos[idx]

def main(
        data_id,
        epoch = 1,
        batchsize = 16,
        ):

    ### load data ###
    train_dir ='D:/data/normalize/{}/train/'.format(data_id)
    test_dir = 'D:/data/normalize/{}/test/'.format(data_id)

    train_csv = np.load(os.path.join(train_dir,'joint.npy'))
    test_csv = np.load(os.path.join(test_dir,'joint.npy'))
    train_left = np.load(os.path.join(train_dir,'left_img.npy'))
    train_right = np.load(os.path.join(train_dir,'right_img.npy'))
    test_left = np.load(os.path.join(test_dir,'left_img.npy'))
    test_right = np.load(os.path.join(test_dir,'right_img.npy'))

    print(train_csv.shape, train_left.shape)

    train_dataset = dataset(train_left, train_right, train_csv)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, drop_last=True)
    test_dataset = dataset(test_left, test_right, test_csv)
    test_dataloader = DataLoader(test_dataset, batch_size=test_csv.shape[0])
    print("Load Data done")

    # define training logs
    training_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists('D:/logs'):
        os.mkdir('D:/logs')
    if not os.path.exists('D:/logs/stereo'):
        os.mkdir('D:/logs/stereo')
    if not os.path.exists('D:/logs/stereo/{}'.format(training_id)):
        os.mkdir('D:/logs/stereo/{}'.format(training_id))
    if not os.path.exists('logs/stereo/{}/models'.format(training_id)):
        os.mkdir('D:/logs/stereo/{}/models'.format(training_id))

    # load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("use device:", device)
    model = stereo_hiruma_att(device)
    model = model.to(device)
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters())
    #optimizer=torch.optim.Adam(model.parameters())
    loss_coef = [1.0, 0.1, 0.001, 0.0001]

    seed = 42
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    #torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)  # Numpy module.
    #random.seed(seed)  # Python random module.	
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True

    # start learning
    with open('D:/logs/stereo/{}/progress.csv'.format(training_id), 'w') as file:
        header = 'epoch,time,train/pos,train/rec_img,train/pt,test/pos,test_img,test_pt\n'
        file.write(header)
    #Start Learining
    for e in range(epoch):
        if e < 100:
            loss_decay = 0.0001
        if e <=250:
            loss_decay = 0.01
        if e>=250:
            loss_decay = 0.1
        start_time = datetime.datetime.now()
        for train_left, train_right, train_position in train_dataloader:
            model.train()
            train_loss, train_position_loss, train_left_loss, train_right_loss, train_pt_loss = 0., 0., 0., 0., 0.
            train_rnn_hidden = None
            train_position =train_position.to(device).float()
            train_left = train_left.to(device).float()
            train_right = train_right.to(device).float()
            for steps in range(49+3):
                if steps < 3:
                    steps = 0
                else:
                    steps = steps-3
                t_train_position = train_position[:,steps]
                t_train_left = train_left[:,steps]
                t_train_right = train_right[:,steps]

                t_pred_left, t_pred_right, t_train_position, att_map1, att_map2, t_pt,pt, train_rnn_hidden = model(t_train_left, t_train_right, t_train_position, train_rnn_hidden)
                train_pt_loss += loss_decay*torch.mean( torch.square( t_pt - pt ) ) 
                train_position_loss += criterion(train_position[:,steps+1], t_train_position)
                train_left_loss += criterion(train_left[:,steps+1], t_pred_left)
                train_right_loss += criterion(train_right[:,steps+1], t_pred_right)
            train_position_loss /= 52
            train_left_loss /= 52
            train_right_loss /= 52
            train_pt_loss / 52

            train_position_loss *= loss_coef[0]
            train_left_loss *= loss_coef[1]
            train_right_loss *= loss_coef[1]
            train_img_loss = train_left_loss.item() + train_left_loss.item()
            
            train_loss = train_position_loss + train_left_loss + train_right_loss + train_pt_loss
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            finish_time = datetime.datetime.now()
            elapsed_time = finish_time - start_time

        if e % 10 == 0:
                with torch.no_grad():
                    for test_left, test_right, test_position in test_dataloader:
                        model.eval()
                        test_loss, test_position_loss, test_left_loss, test_right_loss, test_pt_loss = 0., 0., 0., 0., 0.
                        test_rnn_hidden = None
                        test_position =test_position.to(device).float()
                        test_left = test_left.to(device).float()
                        test_right = test_right.to(device).float()
                        for steps in range(49+3):
                            if steps < 3:
                                steps = 0
                            else:
                                steps = steps-3
                            t_test_position = test_position[:,steps,:]
                            t_test_left = test_left[:,steps,:]
                            t_test_right = test_right[:,steps,:]

                            t_pred_left, t_pred_right, t_test_position, t_test_attmap1, t_test_attmap2,t_pt, pt, test_rnn_hidden = model(t_test_left, t_test_right, t_test_position, test_rnn_hidden)
                            test_pt_loss += loss_decay*torch.mean( torch.square( t_pt - pt ) ) 
                            test_position_loss += criterion(test_position[:,steps+1], t_test_position)
                            test_left_loss += criterion(test_left[:,steps+1], t_pred_left)
                            test_right_loss += criterion(test_right[:,steps+1], t_pred_right)

                        test_pt_loss /= 52
                        test_position_loss /= 52
                        test_left_loss /= 52
                        test_right_loss /= 52

                        test_position_loss *= loss_coef[0]
                        test_left_loss *= loss_coef[1]
                        test_right_loss *= loss_coef[1]
                        test_img_loss = test_left_loss.item()+test_right_loss.item()

                    w_line = '{},{:.4f},{:.16f},{:.16f},{:.16f},{:.16f},{:.16f},{:.16f}'.format(
                            e,  
                            elapsed_time.total_seconds(),
                            train_position_loss.item(), 
                            train_img_loss,
                            train_pt_loss.item(),
                            test_position_loss.item(), 
                            test_img_loss,
                            test_pt_loss.item()
                        )

                    p_line = '{}, time:{:.4f}, train/pos_loss:{:.8f},train/red_loss:{:.8f}, train/pt_loss:{:.8f}, test/pos_loss:{:.8f}, test/rec_loss:{:.8f}, test/pt_loss:{:.8f}'.format(
                            e,  
                            elapsed_time.total_seconds(),
                            train_position_loss.item(), 
                            train_img_loss,
                            train_pt_loss.item(),
                            test_position_loss.item(), 
                            test_img_loss, 
                            test_pt_loss.item()
                        )
                    print('{} epoch:'.format(epoch), p_line)
                    with open('D:/logs/stereo/{}/progress.csv'.format(training_id), 'a') as f:
                        w_line += '\n'
                        f.write(w_line)

        if e % 50 == 0:
            print('Saving Models...')
            torch.save(model.state_dict(), 'D:/logs/stereo/{}/models/model_{}.pth'.format(training_id,e))
            last_save_time = datetime.datetime.now()

    print('Finished Training.')

    #save model
    print('Saving Models...')
    torch.save(model.state_dict(), 'D:/logs/stereo/{}/models/model_final.pth'.format(training_id))   
    print('Finished training_id: {}'.format(training_id))
        
    #save args
    config = {}
    config['data_id'] = data_id
    config['epoch'] = epoch
    config['batchsize'] = batchsize
    config['training_id'] = training_id
    with open('D:/logs/stereo/{}/train_config.json'.format(training_id), 'w') as f:
        json.dump(config, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_id')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=-1, help='batchsize. if set -1, use all sequences for an epoch.')
    args = parser.parse_args()
    main(
        args.data_id,
        epoch = args.epoch,
        batchsize = args.batchsize,
        )