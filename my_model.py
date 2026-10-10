import sys
sys.path.append('/data01/code/mx/All-in-one')  # 添加路径
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import numpy as np
import torch.nn.functional as nn
from subnetwork import *
import utils.transforms as T
from model.ACmix import *
from model.unet import Unet
from typing import List, Optional, Tuple
import utils.coil_combine as C
from utils.data import transforms as DT
import utils.fftc as FFT

# class dataConsistencyTerm(nn.Module):
#     def __init__(self, noise_lvl=None):
#         super(dataConsistencyTerm, self).__init__()
#         self.noise_lvl = noise_lvl
#         if noise_lvl is not None:
#             self.noise_lvl = torch.nn.Parameter(torch.Tensor([noise_lvl]))
#
#     def perform(self, x, k0, mask, sensitivity):
#         """
#         k    - input in k-space
#         k0   - initially sampled elements in k-space  零填充重建
#         mask - corresponding nonzero location
#         """
#         x = T.complex_multiply(x[..., 0].unsqueeze(1), x[..., 1].unsqueeze(1), sensitivity[..., 0],
#                                sensitivity[..., 1])  ########拆成实部虚部分别进行运算，最开始送进来的x即为论文中对应的m^k
#         k = torch.fft(x, 2, normalized=True)
#         v = self.noise_lvl
#
#         if v is not None:  # noisy case
#             # out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)   ############按本行计算方式，此处的v,其实是论文中对应的lamda，alph实际是设为1
#             out = (1 - mask) * k + mask * (v * k + (1 - v) * k0)  #######跟上边是一样的
#         else:  # noiseless case
#             out = (1 - mask) * k + mask * k0
#
#         #### backward op ###
#         x = torch.ifft(out, 2, normalized=True)
#         Sx = T.complex_multiply(x[..., 0], x[..., 1], sensitivity[..., 0], -sensitivity[..., 1]).sum(dim=1)
#         return Sx
# class dataConsistencyTerm(nn.Module):
#     def __init__(self):
#         super(dataConsistencyTerm, self).__init__()
#
#     def perform(self, x, k0, mask, sensitivity, erromap):
#         """
#         k    - input in k-space
#         k0   - initially sampled elements in k-space  零填充重建
#         mask - corresponding nonzero location
#         """
#         x = T.complex_multiply(x[..., 0].unsqueeze(1), x[..., 1].unsqueeze(1), sensitivity[..., 0],
#                                sensitivity[..., 1])  ########拆成实部虚部分别进行运算，最开始送进来的x即为论文中对应的m^k
#         k = torch.fft.fft2(x, 2, normalized=True)
#         v = erromap.permute(0,2,3,1) # 1 320 320 2
#         v = v.unsqueeze(1).repeat(1, 15, 1, 1, 1)
#
#         if v is not None:  # noisy case
#             out = (1 - mask) * k + mask * (v * k + (1 - v) * k0)  #######跟上边是一样的
#         else:  # noiseless case
#             out = (1 - mask) * k + mask * k0
#
#         #### backward op ###
#         x = torch.ifft(out, 2, normalized=True)
#         Sx = T.complex_multiply(x[..., 0], x[..., 1], sensitivity[..., 0], -sensitivity[..., 1]).sum(dim=1)
#         return Sx

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()

        self.apply_A = nn.Linear(2, 242, bias=True)
        self.apply_B = nn.Linear(242, 2, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.mean(x, dim=[2, 3], keepdim=False)  # 1 2
        temp1 = self.relu(self.apply_A(x))
        y = self.apply_B(temp1)
        return y


class dataConsistencyTerm_N(nn.Module):
    def __init__(self):
        super(dataConsistencyTerm_N, self).__init__()
        # self.FCnet = FCNet().cuda()

    def perform(self, x, k0, mask, sensitivity, erromap):
        """
        k    - input in k-space
        k0   - initially sampled elements in k-space  零填充重建
        mask - corresponding nonzero location
        """
        x = T.complex_multiply(x[..., 0].unsqueeze(1), x[..., 1].unsqueeze(1), sensitivity[..., 0],    # 1 15 320 320 2
                               sensitivity[..., 1])
        x_complex = torch.view_as_complex(x)
        k_complex = torch.fft.fft2(x_complex, dim=(2, 3), norm='ortho')   # 1 15 320 320
        k = torch.stack((k_complex.real, k_complex.imag), dim=-1)         # 1 15 320 320 2

        # v = erromap.permute(0, 2, 3, 1)  # 1 320 320 2
        v = erromap  # 1 2
        c = k.shape[1]
        h = k.shape[2]
        w = k.shape[3]
        v = v.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, c, h, w, 1)  #v = v.unsqueeze(1).repeat(1, 15, 1, 1, 1)

        if v is not None:  # noisy case
            out = (1 - mask) * k + mask * (v * k + (1 - v) * k0)
        else:  # noiseless case
            out = (1 - mask) * k + mask * k0

        x_2 = torch.fft.ifft2(torch.complex(out[..., 0], out[..., 1]), dim=(2, 3), norm="ortho")    # 1 15 320 320
        x = torch.stack((x_2.real, x_2.imag), dim=-1).float()      # 1 15 320 320 2
        Sx = T.complex_multiply(x[..., 0], x[..., 1], sensitivity[..., 0], -sensitivity[..., 1]).sum(dim=1)   # 1 320 320 2 下同

        return Sx




class weightedAverageTerm(nn.Module):
    def __init__(self, para=None):
        super(weightedAverageTerm, self).__init__()
        self.para = para
        if para is not None:
            self.para = torch.nn.Parameter(torch.Tensor([para]))

    def perform(self, cnn, Sx):

        x = self.para * cnn + (1 - self.para) * Sx  #########确定一下SS是否为单位阵
        return x

class network(nn.Module):
    def __init__(self, args):
        super(network, self).__init__()
        self.cascades = args.depths
        conv_blocks = []
        dc_blocks = []
        wa_blocks = []
        self.CFNet = CrossFE(num_in_ch=1, num_out_ch=1, upscale=1, num_block=4, num_feat=32, num_residual_feat=3).cuda()
        self.ACMIX = ACmix(in_planes=2, out_planes=2, head=1)
        self.FC = FCNet().cuda()
        for i in range(self.cascades):

            conv_blocks.append(updataz(args).cuda())
            dc_blocks.append(dataConsistencyTerm_N().cuda())
            wa_blocks.append(weightedAverageTerm(args.wc).cuda())

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dc_blocks = nn.ModuleList(dc_blocks)
        self.wa_blocks = nn.ModuleList(wa_blocks)

    def forward(self, x, k, m, c, s):
        x = x.float()
        s = s.float()
        x_abs = torch.sqrt(x.permute(0, 3, 1, 2)[:,0,:,:]**2 + x.permute(0, 3, 1, 2)[:,1,:,:]**2).unsqueeze(1)
        F_h, F_l = self.CFNet(s, x_abs)
        erromap1 = self.ACMIX(x.permute(0,3,1,2))
        erromap2 = self.FC(erromap1)
        for i in range(self.cascades):
            x_cnn = self.conv_blocks[i](x.permute(0, 3, 1, 2), s, F_h, F_l)
            x_cnn = x_cnn.permute(0, 2, 3, 1)
            Sx = self.dc_blocks[i].perform(x, k, m, c, erromap2)
            x = self.wa_blocks[i].perform(x + x_cnn, Sx)
        return x, erromap1


