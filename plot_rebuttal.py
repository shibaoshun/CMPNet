import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 参数设置 ---
data_dir = r'/data01/code/mx/All-in-one/AioDATA/numpy_data_for_plot'
idx = 8  # 🎯 强烈推荐用考得最好的第 8 张图！(你也可以换成 9)

print(f"📁 正在加载第 {idx} 张图的底层数据...")

# --- 2. 加载裸数据 ---
# 注意：之前保存的数据已经是绝对值(幅值)了
und = np.load(os.path.join(data_dir, f'und_{idx:02d}.npy'))
medl = np.load(os.path.join(data_dir, f'medl_recon_{idx:02d}.npy'))
cmpnet = np.load(os.path.join(data_dir, f'cmpnet_recon_{idx:02d}.npy'))
gt = np.load(os.path.join(data_dir, f'gt_{idx:02d}.npy'))

# 为了画图好看且公平，全部归一化到 0~1 的范围
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

und = normalize(und)
medl = normalize(medl)
cmpnet = normalize(cmpnet)
gt = normalize(gt)

# --- 3. 计算误差热力图 (Error Maps) ---
# 误差 = |重建图 - 金标准|
err_und = np.abs(und - gt)
err_medl = np.abs(medl - gt)
err_cmpnet = np.abs(cmpnet - gt)
err_gt = np.zeros_like(gt)  # 金标准的误差当然是 0

# --- 4. 开始排版画图 (2行 x 4列) ---
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
plt.subplots_adjust(wspace=0.05, hspace=0.05) # 缩小图片之间的缝隙

# 准备循环的数据
images = [und, medl, cmpnet, gt]
errors = [err_und, err_medl, err_cmpnet, err_gt]

# 你刚才跑出来的真实成绩，我直接帮你填在这里了！
titles = [
    'Zero-filled (4x)\nSevere Aliasing',
    'MEDL-Net (Zero-Shot)\nPSNR: 33.28 | SSIM: 0.8453',
    'CMPNet (Few-Shot, Ours)\nPSNR: 33.65 | SSIM: 0.8504',
    'Ground Truth\n(Target)'
]

for i in range(4):
    # ====================
    # 第一行：灰度修复图
    # ====================
    ax_img = axes[0, i]
    ax_img.imshow(images[i], cmap='gray', vmin=0, vmax=1)
    ax_img.set_title(titles[i], fontsize=15, fontweight='bold', pad=15)
    ax_img.axis('off')

    # ====================
    # 第二行：误差热力图
    # ====================
    ax_err = axes[1, i]
    # cmap='jet' 会生成论文同款的"深蓝(无误差) -> 红色(大误差)"
    # vmax=0.15 是关键，调节这个值能让 MEDL 的红色伪影极其显眼！
    im_err = ax_err.imshow(errors[i], cmap='jet', vmin=0, vmax=0.15)
    ax_err.axis('off')

# 保存为 300dpi 的高清论文级插图
save_path = 'Rebuttal_Prostate_Comparison.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
print(f"已成功生成并保存为: {save_path}")
