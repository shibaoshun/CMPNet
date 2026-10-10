# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
import torch
from trainer_utils import evaluation, model_selection


parser = argparse.ArgumentParser()
# parser.add_argument('--result_path', default=r"lr2e4_lamda0.1_beta0.5/backbone/", type=str, help='path to save results')
parser.add_argument('--result_path', default=r"result468/backbone/", type=str, help='path to save results')

# %%Training config
parser.add_argument('--net_name', default='backbone', type=str, help='model')
parser.add_argument("--center_fraction", type=float, default=0.08)
parser.add_argument("--wc", type=float, default=0.5)
parser.add_argument('--depths', default=3, type=int, help='output channel of network')
parser.add_argument('--batch_size', default=1, type=int, help='batch_size for training')
# %%
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--num_workers', default=0, type=int, help='num_workers for data loader')
parser.add_argument('--device', type=str, default='1', help='cuda device')
args = parser.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# CUDA_VISIBLE_DEVICES controlled by shell

if __name__ == '__main__':

    device = args.device
    model = model_selection(args.net_name, args)

    model = model.cuda()
    result_path = args.result_path
    evaluation(args, model, result_path)


