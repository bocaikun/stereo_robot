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
            nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=1),
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
        #left = self.coor(left)
        #right = self.coor(right)
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

    def get_positional_encodings(self, img):
        batchsize, channel, height, width  = img.shape
        x_linear = torch.linspace(0., 1., width).tile((height, 1))
        y_linear = torch.linspace(0., 1., height).tile((width, 1)).T
        positional_encodings = torch.cat(
            [el.reshape(-1, 1) for el in [x_linear, y_linear]], axis=-1)
        pt_value = positional_encodings.expand(batchsize, channel, *positional_encodings.shape) # shape(batch, c, h*w, 2)
        return pt_value.to(img.device)

    def get_att_map(self,img, temp=0.0001):
        batchsize, channel, height, width  = img.shape
        map = img.view(batchsize, channel, height*width)
        softmax_out = F.softmax(map/temp, dim=-1)
        att_map = softmax_out.unsqueeze(-2)
        return att_map
    
    def get_pt(self, att_map, pt_value):
        batchsize, channel, _, hw  = att_map.shape
        pt = torch.matmul(att_map, pt_value) # shape(batch, c, 1, 2)
        pt = pt.reshape(batchsize, channel*2)
        return pt

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
        
        pt_value = self.get_positional_encodings(out)
        att_map = self.get_att_map(out,temp=0.0001)  # shape(bacth, c, 1, h*w)
        pt = self.get_pt(att_map, pt_value)
        return out, pt

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
            nn.Linear(6, 12), nn.ReLU(),
            nn.Linear(12, output_dim), nn.ReLU(),
        )
    def forward(self, x):
        x = self.fnn(x)
        return x 

class from_pt(nn.Module):
    def __init__(self,output_dim):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, output_dim), nn.ReLU(),
        )
    def forward(self, x):
        x = self.fnn(x)
        return x 

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
            nn.Linear(input_dim, 12), nn.ReLU(),
            nn.Linear(12, 6)
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
    def __init__(self, img_fusion_dim=24,pos_fusion_dim=12):
        super(stereo_cae_att, self).__init__()
        self.io_dim = 32*28*28
        self.encoder = encoder()
        self.att = non_local(dim=32)
        self.to_fusion = to_fusion(input_dim=self.io_dim, output_dim=img_fusion_dim)
        self.to_position = to_position(output_dim=pos_fusion_dim)
        self.from_pt = from_pt(output_dim=8)
        self.predrnn = nn.LSTMCell(12+8+24, 12+24)
        self.position = position(input_dim=pos_fusion_dim)
        self.to_decoder = to_decoder(input_dim=img_fusion_dim, output_dim=self.io_dim)
        self.decoder = decoder()

    def forward(self, left, right, now_position, feat_hc):
        img_fusion = self.encoder(left, right)
        img_fusion, pt = self.att(img_fusion)
        img_fusion = torch.flatten(img_fusion, start_dim=1)
        img_fusion = self.to_fusion(img_fusion)
        
        pt = self.from_pt(pt)
        pos = self.to_position(now_position)

        now_fusion = torch.cat([pos, pt, img_fusion], dim=-1)
        feat_hid, feat_state = self.predrnn(now_fusion, feat_hc)
        
        position = self.position(feat_hid[:, :12])
        T_fusion = self.to_decoder(feat_hid[:,12:])
        T_fusion = T_fusion.view(-1, 32, 28, 28)
        T_left, T_right = torch.split(T_fusion, [16, 16], dim=1)
        T_left, T_right = self.decoder(T_left, T_right)
        return T_left, T_right, position, (feat_hid, feat_state)