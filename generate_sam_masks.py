import sigpy as sp
import sigpy.mri as mr
import os
import h5py
import numpy as np
from scipy.io import savemat

# ================= 配置区 =================
# 1. 你的 fastMRI 前列腺原始 .h5 文件所在的文件夹
SOURCE_H5_DIR = r'/data01/code/mx/All-in-one/AioDATA'

# 2. 你想把生成好的 .mat 文件存到哪里（目标根目录）
TARGET_MAT_DIR = r'/data01/code/mx/All-in-one/AioDATA/fastmri_prostate_h5_files'

# 3. 数据集划分需求（你需要 20张训练, 10张验证, 10张测试）
SPLIT_NUMS = {'train': 20, 'val': 10, 'test': 10}
# ==========================================

def create_dirs(base_path):
    """自动创建规范的文件夹结构"""
    folders = ['train', 'train_cms', 'val', 'val_cms', 'test', 'test_cms']
    for folder in folders:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)
    print(f"文件夹结构已在 {base_path} 创建完毕！")

def extract_and_save():
    create_dirs(TARGET_MAT_DIR)
    h5_files = [f for f in os.listdir(SOURCE_H5_DIR) if f.endswith('.h5')]
    
    saved_counts = {'train': 0, 'val': 0, 'test': 0}
    current_phase = 'train'
    slice_idx = 0 

    for h5_file in h5_files:
        file_path = os.path.join(SOURCE_H5_DIR, h5_file)
        
        with h5py.File(file_path, 'r') as f:
            kspace_volume = f['kspace'][()] 
            
            # ==========================================
            # 🚨 核心修复 1：强行降维！
            # 如果读出来是 5 维 (1, 30, 14, 200, 150)，我们强行脱掉最外面那层壳
            # 把它变成标准的 4 维 (切片数, 线圈数, 高, 宽)
            # ==========================================
            while len(kspace_volume.shape) > 4:
                kspace_volume = kspace_volume[0]
                
            num_slices = kspace_volume.shape[0]
            
            for i in range(num_slices):
                if saved_counts['test'] >= SPLIT_NUMS['test']:
                    print("🎉 40 张切片全部提取完毕！")
                    return
                
                if saved_counts['train'] < SPLIT_NUMS['train']:
                    current_phase = 'train'
                elif saved_counts['val'] < SPLIT_NUMS['val']:
                    current_phase = 'val'
                else:
                    current_phase = 'test'
                
                # 现在单张切片绝对是标准的 3D 了：(14, 200, 150)
                single_slice_kspace = kspace_volume[i]
                
                print(f"正在计算第 {slice_idx+1}/40 张真实灵敏度图... 当前形状: {single_slice_kspace.shape}")
                
                try:
                    # ==========================================
                    # 🚨 核心修复 2：去掉报错的 Device 参数，直接让 CPU 秒算
                    # ==========================================
                    real_maps = mr.app.EspiritCalib(single_slice_kspace).run()
                    real_maps = np.complex64(real_maps)
                except Exception as e:
                    print(f"计算灵敏度图时出错: {e}，将跳过此层。")
                    continue
                
                kspace_filename = f"prostate_slice_{slice_idx:03d}.mat"
                maps_filename = f"prostate_slice_{slice_idx:03d}.mat"
                
                kspace_save_path = os.path.join(TARGET_MAT_DIR, current_phase, kspace_filename)
                maps_save_path = os.path.join(TARGET_MAT_DIR, f"{current_phase}_cms", maps_filename)
                
                savemat(kspace_save_path, {'kspace': single_slice_kspace})
                savemat(maps_save_path, {'maps': real_maps})
                
                saved_counts[current_phase] += 1
                slice_idx += 1
                
                print(f"✅ 已保存: {kspace_filename} -> {current_phase} 集合")

if __name__ == '__main__':
    extract_and_save()
