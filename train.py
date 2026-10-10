import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
import random
import numpy as np
import torch
from trainer_utils import model_selection, trainer

# %%data path config
parser = argparse.ArgumentParser()
parser.add_argument('--result_path', default=r"result468/", type=str, help='path to save results')

# %%Training config
parser.add_argument('--net_name', default='backbone', type=str, help='model')
parser.add_argument("--center_fraction", type=float, default=0.08)
parser.add_argument("--wc", type=float, default=0.5)
parser.add_argument('--depths', default=3, type=int, help='output channel of network')
parser.add_argument('--max_epochs', default=50, type=int, help='maximum epoch number for training')
parser.add_argument('--batch_size_train', default=1, type=int, help='batch_size for training')
parser.add_argument('--batch_size_eval', default=1, type=int, help='batch_size for validation and test')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
parser.add_argument("--lr", type=float, default=0.0002, help="Initial learning rate")
parser.add_argument("--milestone", type=int, default=[30], help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr_decay", type=float, dest="lr_decay", help="ADAM Learning rate decay (on step)", default=0.1)
# %%
parser.add_argument('--resume_training', default=0, type=int, help='1: resume training: load model from "current.pth"')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--num_workers', default=2, type=int, help='num_workers for data loader')
parser.add_argument('--device', type=str, default='1', help='cuda device')
# %%
args = parser.parse_args()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
# %%
if __name__ == '__main__':
    net_name = args.net_name
    model = model_selection(net_name, args)
    model = model.cuda()
    save_path = net_name.replace(' ', '')
    save_path = args.result_path + save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trainer(args, model, save_path)
