import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from trainer_utils import model_selection, psnr, ssim, nmse # 引用你现有的指标计算工具
from dataloader import MRIDataset_p, load_train_data
import utils.transforms as T
from model.SAM import IntroduceSAM

# --- 参数配置 ---
parser = argparse.ArgumentParser()
parser.add_argument('--result_path', default=r"result_finetune_prostate/", type=str)
parser.add_argument('--net_name', default='backbone', type=str)
parser.add_argument("--center_fraction", type=float, default=0.08)
parser.add_argument("--wc", type=float, default=0.5)
parser.add_argument('--depths', default=3, type=int)
parser.add_argument('--device', type=str, default='5', help='用你刚才空闲的显卡')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

def test_and_visualize(args, model, save_path):
    print('=============================================')
    print('前列腺 10张测试集评估与可视化')
    print('=============================================')
    
    # 1. 强制进入绝对的考试模式 (禁止学习)
    model.eval()
    model_sam = IntroduceSAM().cuda()
    model_sam.eval()
    
    # 2. 加载你刚才微调出来的最高分权重！
    best_weight_path = os.path.join("result468", "backbone", "best.pth")
    if not os.path.exists(best_weight_path):
        print("❌ 找不到微调好的权重 best_finetuned.pth，是不是还没训练完？")
        return
    model.load_state_dict(torch.load(best_weight_path))
    print("✅ 成功加载前列腺最强微调权重！")

    # 3. 准备专门保存对比图的文件夹
    image_save_dir = os.path.join(save_path, 'comparison_images')
    os.makedirs(image_save_dir, exist_ok=True)

    # 4. 只加载剩下的 10 张绝对陌生的 Test 数据
    dataset_dir_p = r'/data01/code/mx/All-in-one/AioDATA/fastmri_prostate_h5_files'
    import glob
    # 直接去 test 和 test_cms 文件夹里抓取那 10 张图并配对！
    test_kspace_files = sorted(glob.glob(os.path.join(dataset_dir_p, 'test', '*.mat')))
    test_maps_files = sorted(glob.glob(os.path.join(dataset_dir_p, 'test_cms', '*.mat')))
    test_data_list = list(zip(test_kspace_files, test_maps_files))
    
    Dataset_test_p = MRIDataset_p(test_data_list, acceleration=4, center_fraction=args.center_fraction)
    testloader = DataLoader(Dataset_test_p, shuffle=False, batch_size=1, num_workers=0)

    test_psnr_list, test_ssim_list, test_nmse_list = [], [], []
    
    print(f"📁 准备开始测试，共有 {len(testloader)} 张测试图像...")
    
    for count, samples in enumerate(testloader):
        img_und, img_gt, rawdata_und, masks, sensitivity, filename = samples
        
        img_gt = img_gt.clone().detach().cuda()
        img_und = img_und.clone().detach().cuda()
        rawdata_und = rawdata_und.clone().detach().cuda()
        masks = masks.clone().detach().cuda()
        sensitivity = sensitivity.clone().detach().cuda()
        
        zf = T.complex_abs(img_und).data.to('cpu')
        
        # 模型推理
        with torch.no_grad():
            sam_map_test = model_sam(zf.unsqueeze(0))
            rec, _ = model(img_und, rawdata_und, masks, sensitivity, sam_map_test)
        
        # 转换为人类可看的 numpy 图像幅值
        sense_recon = T.complex_abs(rec).data.to('cpu').numpy()[0] # CAPNet 修复的高清图
        sense_gt = T.complex_abs(img_gt).data.to('cpu').numpy()[0] # 真实的完美金标准图
        sense_und = T.complex_abs(img_und).data.to('cpu').numpy()[0] # 机器采样的严重重影图
        
        # 计算三大金标准指标
        test_psnr = psnr(abs(sense_gt), abs(sense_recon))
        test_ssim = ssim(abs(sense_gt), abs(sense_recon))
        test_nmse = nmse(abs(sense_gt), abs(sense_recon))
        save_data_dir = r'/data01/code/mx/All-in-one/AioDATA/numpy_data_for_plot'

        os.makedirs(save_data_dir, exist_ok=True)
        
        # 4. 保存 CMPNet (微调后) 修复的图
        np.save(os.path.join(save_data_dir, f'cmpnet_recon_{count+1:02d}.npy'), abs(sense_recon))
        test_psnr_list.append(test_psnr)
        test_ssim_list.append(test_ssim)
        test_nmse_list.append(test_nmse)
        
        print(f"图 {count+1:02d} | PSNR: {test_psnr:.2f} dB | SSIM: {test_ssim:.4f} | NMSE: {test_nmse:.4f}")
        
        # ==========================================
        # 🌟 核心：画图并保存，直接用于论文排版
        # ==========================================
        plt.figure(figsize=(15, 5)) # 设置一张长条形的画布
        
        # 第一张图：未修复的烂图 (Zero-filled)
        plt.subplot(1, 3, 1)
        plt.imshow(abs(sense_und), cmap='gray')
        plt.title('Undersampled (Before Recon)', fontsize=14)
        plt.axis('off')
        
        # 第二张图：CAPNet 修复的图
        plt.subplot(1, 3, 2)
        plt.imshow(abs(sense_recon), cmap='gray')
        plt.title(f'CAPNet Recon\nPSNR: {test_psnr:.2f} | SSIM: {test_ssim:.4f}', fontsize=14)
        plt.axis('off')
        
        # 第三张图：真实的金标准图 (Ground Truth)
        plt.subplot(1, 3, 3)
        plt.imshow(abs(sense_gt), cmap='gray')
        plt.title('Ground Truth (Target)', fontsize=14)
        plt.axis('off')
        
        # 保存图片到指定文件夹
        save_name = os.path.join(image_save_dir, f'test_result_{count+1:02d}.png')
        plt.savefig(save_name, bbox_inches='tight', dpi=300) # dpi=300 保证论文打印高清
        plt.close() # 画完一定要关掉，不然显存/内存会爆

    # 最终成绩单核算
    print('=============================================')
    print(f"🌟 测试集 10 张图最终平均成绩：")
    print(f"Mean PSNR: {np.mean(test_psnr_list):.4f} dB")
    print(f"Mean SSIM: {np.mean(test_ssim_list):.4f}")
    print(f"Mean NMSE: {np.mean(test_nmse_list):.4f}")
    print(f"📁 对比高清图已保存在: {image_save_dir}")
    print('=============================================')

if __name__ == '__main__':
    model = model_selection(args.net_name, args)
    model = model.cuda()
    test_and_visualize(args, model, args.result_path)
