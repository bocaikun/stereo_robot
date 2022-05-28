import torch
from torch import nn, softmax
import numpy as np
import torch.nn.functional as F
from einops import rearrange

class AddCoords(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)


        return ret

class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.coor = AddCoords()
        self.features = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.embedding = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, left, right):
        left = self.coor(left)
        right = self.coor(right)
        left = self.features(left)
        right = self.features(right)
        fusion = torch.cat([left, right], dim=1)
        fusion = self.embedding(fusion)
        return fusion

class non_local(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        # self.out_linear = nn.Linear(head*dim, dim)
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.out_kqv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        )
        
    def forward(self, x):
        k = self.to_k(x)
        q = self.to_q(x)
        v = self.to_v(x)
        
        k = rearrange(k, 'b c h w -> b (h w) c')
        q = rearrange(q, 'b c h w -> b (h w) c')
        v = rearrange(v, 'b c h w -> b (h w) c')

        dots = torch.einsum('bic,bjc->bij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bij,bkc->bic', attn, v)
        out = rearrange(out, 'b (h w) c -> b c h w', h=28, w=28)
        out = self.out_kqv(out)
        return out

class to_fusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, 1024),nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, output_dim),nn.ReLU(),
        )

    def forward(self, x):
        x = self.fnn(x)
        return x 

class to_position(nn.Module):
    def __init__(self,output_dim):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(6, 24), nn.ReLU(),
            nn.Linear(24, 24), nn.ReLU(),
        )


class to_decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 1024), nn.ReLU(),
            nn.Linear(1024, output_dim),nn.ReLU(),
        )

    def forward(self, x):
        x = self.fnn(x)
        return x

class position(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, 24), nn.ReLU(),
            nn.Linear(24, 6)
        )

    def forward(self, x):
        x = self.fnn(x)
        return x

class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_img = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1,  padding=1),nn.ReLU(),
            nn.ConvTranspose2d(32, 8, kernel_size=3, stride=2,output_padding=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=5, stride=2,output_padding=1, padding=2), 
        )
    
    def forward(self, left, right):
        left = self.to_img(left)
        right = self.to_img(right)
        return left, right

class stereo_cae_att(nn.Module):
    def __init__(self, img_fusion_dim=24,pos_fusion_dim=24):
        super(stereo_cae_att, self).__init__()
        self.io_dim = 32*28*28
        self.encoder = encoder()
        self.att = non_local(dim=32)
        self.to_fusion = to_fusion(input_dim=self.io_dim, output_dim=img_fusion_dim)
        self.to_position = to_position(output_dim=pos_fusion_dim)
        self.predrnn = nn.LSTMCell(24+24, 24+24)
        self.position = position(input_dim=24)
        self.to_decoder = to_decoder(input_dim=img_fusion_dim, output_dim=self.io_dim)
        self.decoder = decoder()

    def forward(self, now_position, left, right, feat_hc):
        img_fusion = self.encoder(left, right)
        img_fusion = self.att(img_fusion)
        img_fusion = torch.flatten(img_fusion, start_dim=1)
        img_fusion = self.to_fusion(img_fusion)
        pos_fusion = self.to_position(now_position)

        now_fusion = torch.cat([pos_fusion, img_fusion], dim=-1)
        feat_hid, feat_state = self.predrnn(now_fusion, feat_hc)
        
        position = self.to_position(feat_hid[:, :24])
        T_fusion = self.to_decoder(feat_hid[:,24:])
        T_fusion = T_fusion.view(-1, 32, 28, 28)
        T_left, T_right = torch.split(T_fusion, [16, 16], dim=1)
        T_left, T_right = self.decoder(T_left, T_right)
        return position, T_left, T_right, (feat_hid, feat_state)