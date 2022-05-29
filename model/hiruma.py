import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hiruma(nn.Module):
    def __init__(self, device):
        super(Hiruma, self).__init__()
        self.device = device
        self.q_num = 4
        self.k_dim = 10
        self.q_dim = 10
        self.f_dim = 10

        kds = [3, 30, 30, self.k_dim*self.q_num]
        fds = [3, 30, 30, self.f_dim*self.q_num]
        ks = [5, 3, 3, 1]
        sts = [2, 2, 1, 1]

        self.conv = nn.Sequential(
                        nn.Conv2d(kds[0], kds[1], kernel_size=ks[0], stride=sts[0]), nn.ReLU(),
                        nn.Conv2d(kds[1], kds[2], kernel_size=ks[1], stride=sts[1]), nn.ReLU(),
                        nn.Conv2d(kds[2], kds[3], kernel_size=ks[2], stride=sts[2]), nn.ReLU(),
                      )

        self.k_conv = nn.Conv2d(kds[3], kds[3], kernel_size=1, stride=1, bias=False)
                        
        self.f_conv = nn.Conv2d(kds[3], kds[3], kernel_size=1, stride=1, bias=False)

        self.query = torch.nn.Parameter(torch.randn((self.q_num, self.q_dim), dtype=torch.float, device=self.device), requires_grad=True)
        nn.init.kaiming_uniform_(self.query, a=np.sqrt(5), mode='fan_in')

    def __call__(self, in_img):
        key, query, pt_value, feat_value = self.get_kqv(in_img)
        pt_att_map = self.get_attention_map(key, query)
        pt_out, feat_out = self.get_att_out(pt_att_map, pt_value, feat_value)

        pt_out = pt_out.reshape(-1, self.q_num*2)
        feat_out = feat_out.reshape(-1, self.f_dim*self.q_num)
        return pt_out, feat_out, pt_att_map

    def get_kqv(self, in_img):
        in_img = self.conv(in_img)
        key = self.k_conv(in_img)
        key = key.permute(0, 2, 3, 1)
        batch_size, key_h, key_w = key.shape[:3]
        key = key.view(*key.shape[:3], self.q_dim, self.q_num)

        batch_size, key_h, key_w = key.shape[:3]
        query = self.query.expand(key_h, key_w, batch_size, self.q_num, self.k_dim)
        query = query.permute(2, 0, 1, 3, 4)

        pt_value_el = self.get_positional_encodings(key_h, key_w)
        pt_value = pt_value_el.expand(batch_size, self.q_num, *pt_value_el.shape)
        feat_value = self.f_conv(in_img)
        feat_value = feat_value.view(feat_value.shape[0], self.f_dim, self.q_num, feat_value.shape[2]*feat_value.shape[3])
        feat_value = feat_value.permute(0, 2, 3, 1)

        # key: (batch_size, key_h, key_w, k_dim, q_num)
        # query: (batch_size, ch, key_h, key_w, q_num, q_dim)
        # pt_value: (batch_size, q_num, key_h*key_w, 2)
        # feat_value: (batch_size, q_num, key_h*key_w, f_dim)
        return key, query, pt_value, feat_value

    def get_attention_map(self, key, query):
        batch_size, key_h, key_w, _, _ = key.shape
        scaling_constant = torch.sqrt(torch.tensor(key_h*key_w, dtype=torch.float, requires_grad=False, device=key.device))

        q_mul_k = torch.matmul(query, key)
        q_mul_k = torch.div(q_mul_k, scaling_constant)
        q_mul_k = torch.mul(q_mul_k, torch.eye(self.q_num, dtype=torch.float, device=key.device))
        q_mul_k = torch.sum(q_mul_k, dim=-1)

        q_mul_k = q_mul_k.permute(0, 3, 1, 2)
        q_mul_k = q_mul_k.view(batch_size, self.q_num, key_h*key_w)

        att_map = F.softmax(q_mul_k/1, dim=-1)
        att_map = att_map.unsqueeze(-2)

        # att_map: (batch_size, q_num, 1, key_h*key_w)
        return att_map

    def get_att_out(self, att_map, pt_value, feat_value):
        pt_out = torch.matmul(att_map, pt_value)
        feat_out = torch.matmul(att_map, feat_value)
        return pt_out, feat_out

    def get_positional_encodings(self, height, width):
        x_linear = torch.linspace(0, 1.0, width ).tile((height, 1))
        y_linear = torch.linspace(0, 1.0, height).tile((width, 1)).T
        return torch.cat([el.reshape(-1, 1) for el in [x_linear, y_linear]], axis=-1).to(self.device)
