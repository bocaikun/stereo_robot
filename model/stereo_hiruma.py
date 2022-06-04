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
        self.fnn = nn.Linear(16, 16, bias=False)
    def forward(self, x):
        x = self.fnn(x)
        return x

class feat_embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.fnn = nn.Linear(80, 80, bias=False)
    def forward(self, x):
        x = self.fnn(x)
        return x

class query_embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.fnn = nn.Linear(80, 80, bias=False)
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

class stereo_hiruma_att(nn.Module):
    def __init__(self, device):
        super(stereo_hiruma_att, self).__init__()
        imgsize = 40*24*24
        self.att = Hiruma(device)
        self.pt_embedding = pt_embedding()
        self.feat_embedding = feat_embedding()
        self.query_embedding = query_embedding()
        self.predrnn = nn.LSTMCell(80+6+16, 6+16)
        self.to_decoder = to_decoder(input_dim=40, output_dim=imgsize)
        self.decoder = decoder()
        self.position = position(input_dim=6)
        sub_im_size = 24
        
        grid = np.meshgrid( np.linspace(0., 1., num=sub_im_size, dtype=np.float32),
                            np.linspace(0., 1., num=sub_im_size, dtype=np.float32),
                            indexing='ij')
        self.i_map = torch.Tensor( np.reshape(grid[0], [1, 1, sub_im_size, sub_im_size]) )
        self.j_map = torch.Tensor( np.reshape(grid[1], [1, 1, sub_im_size, sub_im_size]) )
        self.grid_stacked = torch.Tensor( np.stack( [grid] * 40, axis=0) )
      
        self.i_map = self.i_map.cuda(0)
        self.j_map = self.j_map.cuda(0)
        self.grid_stacked = self.grid_stacked.cuda(0)


    def generate_heatmap(self, ij):
        scale = 1.0
        _xxx, _yyy = torch.split(ij, 4, dim=-1)
        _xxx = _xxx.repeat(1,10)
        _yyy = _yyy.repeat(1,10)

        #_xxx= ij[:,:self.k_dim]
        #_yyy= ij[:,self.k_dim:]

        __ij= torch.concat([torch.unsqueeze(_xxx, -1), torch.unsqueeze(_yyy, -1)], axis=-1)

        ij = __ij.reshape( [-1, 40, 2, 1, 1] )
        squared_distances = torch.sum( torch.pow(ij - self.grid_stacked, 2.0), axis=2 )
        heatmap = scale * torch.exp(-squared_distances / 0.1)

        return heatmap

    def forward(self, left, right, now_position, feat_hc):
        
        img1, pt_out1, feat_out1, att_map1 = self.att(left)
        img2, pt_out2, feat_out2, att_map2 = self.att(right)

        pt_fusion = torch.cat([pt_out1, pt_out2], dim=-1)
        #pt_fusion = self.pt_embedding(pt_fusion)
        feat_fusion = torch.cat([feat_out1, feat_out2], dim=-1)
        feat_fusion = self.feat_embedding(feat_fusion)

        # query1 = query1.view(-1, 4*10)
        # query2 = query2.view(-1, 4*10)
        # query_fusion = torch.cat([query1, query2],dim=-1)
        # query_fusion = self.query_embedding(query_fusion)

        now_fusion = torch.cat([feat_fusion, now_position, pt_fusion], dim=-1) #80+16+80+6
        feat_hid, feat_state = self.predrnn(now_fusion, feat_hc) #80+6

        # n_left_feat = feat_hid[:, :40]
        # n_right_feat = feat_hid[:, 40:80]
        n_position = self.position(feat_hid[:, :6])
        n_pt = feat_hid[:, 6:]

        heat_left = self.generate_heatmap(n_pt[:,:8])
        heat_right = self.generate_heatmap(n_pt[:,8:])
        
        #n_left_feat = self.to_decoder(n_left_feat)
        #n_left_feat = n_left_feat.view(-1, 40, 24, 24)
        n_left_feat = torch.mul(heat_left, img1)
        n_left_feat = self.decoder(n_left_feat)
        #n_right_feat = self.to_decoder(n_right_feat)
        #n_right_feat = n_right_feat.view(-1, 40, 24, 24)
        n_right_feat = torch.mul(heat_right, img2)
        n_right_feat = self.decoder(n_right_feat)

        return n_left_feat, n_right_feat, n_position, att_map1, att_map2, n_pt, pt_fusion, (feat_hid, feat_state)