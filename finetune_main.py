import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
import torch
from trainer_utils import trainer_finetune, model_selection

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', default=r"result_finetune_prostate/", type=str, help='保存微调结果的路径')
parser.add_argument('--net_name', default='backbone', type=str, help='model')
parser.add_argument("--center_fraction", type=float, default=0.08)
parser.add_argument("--wc", type=float, default=0.5)
parser.add_argument('--depths', default=3, type=int)
parser.add_argument('--batch_size_train', default=1, type=int)
parser.add_argument('--batch_size_eval', default=1, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--device', type=str, default='1', help='cuda device')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

if __name__ == '__main__':
    # 自动创建保存权重的文件夹
    os.makedirs(args.result_path, exist_ok=True)
    
    # 实例化模型
    model = model_selection(args.net_name, args)
    model = model.cuda()
    
    # 启动微调炼丹炉！
    trainer_finetune(args, model, args.result_path)
