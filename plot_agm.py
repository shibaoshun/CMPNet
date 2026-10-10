import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ==========================================
# 🌟 全局字体配置：学术罗马体 + 防报错机制
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['font.size'] = 14

def plot_agm_features():
    # 刚刚跑出来的特征文件
    files = [
        {'path': 'feat_brain.pt', 'title': 'Brain 4x'},
        {'path': 'feat_knee.pt',  'title': 'Knee 6x'},
    ]
    
    # 创建 2行 x 3列 的画板，适当加宽以便在右侧放刻度尺
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # 🎯 极限压缩间距，让图片背靠背贴合
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    
    for row, info in enumerate(files):
        # 检查文件是否存在
        if not os.path.exists(info['path']):
            print(f"⚠️ 找不到文件: {info['path']}，请检查是否在当前目录下！")
            continue
            
        print(f"正在处理 {info['title']} 的可视化...")
        data = torch.load(info['path'])
        
        # 1. 提取输入图和金标准
        zf_img = np.abs(data['zf'].squeeze().numpy())
        gt_img = np.abs(data['gt'].squeeze().numpy())
            
        # 2. 提取 AGM 特征图并融合通道
        agm_feat = data['agm_feat'].squeeze() 
        if agm_feat.ndim == 3:
            feat_2d = torch.mean(torch.abs(agm_feat), dim=0).numpy()
        else:
            feat_2d = np.abs(agm_feat.numpy())
            
        # 🎯【关键细节】为了让特征图能完美使用 0~0.200 的刻度尺，
        # 我们将其归一化，使其最高亮度对应 0.200，保证热力图颜色饱满
        feat_2d = (feat_2d / (feat_2d.max() + 1e-8)) * 0.200
        
        # 3. 计算真实的伪影误差 (Ground Truth Error Map)
        error_map = np.abs(zf_img - gt_img)
        
        # ================= 开始画图 =================
        
        # --- 第一列：Zero-filled 输入图 ---
        ax1 = axes[row, 0]
        ax1.imshow(zf_img, cmap='gray')
        ax1.axis('off')
        if row == 0: 
            ax1.set_title("Zero-Filled Input", fontsize=18, fontweight='bold', pad=15)
        # 左侧行标签 (Brain 4x / Knee 6x)
        ax1.text(-0.05, 0.5, info['title'], transform=ax1.transAxes, 
                 fontsize=18, fontweight='bold', va='center', ha='right', rotation=90)
                 
        # --- 第二列：AGM 特征热力图 ---
        ax2 = axes[row, 1]
        # 严格限制 0~0.200 的范围，对齐刻度尺
        ax2.imshow(feat_2d, cmap='jet', vmin=0, vmax=0.200) 
        ax2.axis('off')
        if row == 0: 
            ax2.set_title("AGM Spatial Feature", fontsize=18, fontweight='bold', pad=15)
        
        # --- 第三列：真实的伪影误差图 ---
        ax3 = axes[row, 2]
        # 严格限制 0~0.200 的范围，对齐刻度尺
        ax3.imshow(error_map, cmap='jet', vmin=0, vmax=0.200) 
        ax3.axis('off')
        if row == 0: 
            ax3.set_title("Ground Truth Artifact", fontsize=18, fontweight='bold', pad=15)

    # ================= 绘制金标准刻度尺 =================
    # 在整个图表的最右侧（热力图旁边）画一条统一的刻度尺
    norm = Normalize(vmin=0, vmax=0.200)
    sm = ScalarMappable(norm=norm, cmap='jet')
    
    # 自动从第2列和第3列偷一点空间来放刻度尺，完美居中对齐
    cbar = fig.colorbar(sm, ax=axes[:, 1:], shrink=0.85, aspect=25, pad=0.03)
    
    # 精确设定金标准刻度
    ticks = [0.000, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(['0.000', '0.025', '0.050', '0.075', '0.100', '0.125', '0.150', '0.175', '0.200'])
    cbar.ax.tick_params(labelsize=14) 

    # 存为 300 dpi 的高清图片
    save_path = 'agm_features_pro.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n🎉 大功告成！带金标准刻度尺的无缝排版大图已保存为: {save_path}")

if __name__ == '__main__':
    plot_agm_features()
