import torch
from torch import nn, softmax
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from model.hiruma import Hiruma


class position(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, 24), nn.ReLU(),
            nn.Linear(24, 24), nn.ReLU(),
            nn.Linear(24, 6)
        )
    def forward(self, x):
        x = self.fnn(x)
        return x

class pt_embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.fnn = nn.Linear(8, 8, bias=False)
    def forward(self, x):
        x = self.fnn(x)
        return x

class feat_embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.fnn = nn.Linear(40, 40, bias=False)
    def forward(self, x):
        x = self.fnn(x)
        return x

class to_decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, output_dim),nn.ReLU(),
        )
    def forward(self, x):
        x = self.fnn(x)
        return x

class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_img = nn.Sequential(
            nn.ConvTranspose2d(40, 30, kernel_size=3, stride=1),nn.ReLU(),
            nn.ConvTranspose2d(30, 30, kernel_size=3, stride=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(30, 3, kernel_size=5, stride=2, output_padding=1) 
        )
    
    def forward(self, img):
        img = self.to_img(img)
        return img

class single_hiruma_att(nn.Module):
    def __init__(self, device):
        super(single_hiruma_att, self).__init__()
        imgsize = 40*24*24
        self.att = Hiruma(device)
        self.pt_embedding = pt_embedding()
        self.feat_embedding = feat_embedding()
        self.predrnn = nn.LSTMCell(8+40+6, 40+6)
        self.to_decoder = to_decoder(input_dim=40, output_dim=imgsize)
        self.decoder = decoder()
        self.position = position(input_dim=6)


    def forward(self, left, now_position, feat_hc):
        
        pt_out1, feat_out1, att_map1 = self.att(left)
        pt_fusion = self.pt_embedding(pt_out1)
        feat_fusion = self.feat_embedding(feat_out1)

        now_fusion = torch.cat([pt_fusion, feat_fusion, now_position], dim=-1) #8+40+6
        feat_hid, feat_state = self.predrnn(now_fusion, feat_hc) #40+6

        n_left_feat = feat_hid[:, :40]
        n_position = self.position(feat_hid[:, 40:])
        
        n_left_feat = self.to_decoder(n_left_feat)
        n_left_feat = n_left_feat.view(-1, 40, 24, 24)
        n_left_feat = self.decoder(n_left_feat)

        return n_left_feat, n_position, att_map1, (feat_hid, feat_state)