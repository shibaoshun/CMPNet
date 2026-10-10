import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ==========================================
# 🌟 终极字体解决方案：无报错的学术罗马体 & 动态大字号
# ==========================================
plt.rcParams['font.family'] = 'serif'
# 自动回退机制：先找Times，找不到就用Linux自带的DejaVu Serif（完美平替，绝不报错）
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
# 基础字号适当提升
plt.rcParams['font.size'] = 14 

# ==========================================
# 1. 基础参数与放大框坐标设置
# ==========================================
data_dir = r'/data01/code/mx/All-in-one/AioDATA/numpy_data_for_plot'
idx =3  

x_min, x_max = 60, 95    
y_min, y_max = 100, 135  

print(f"📁 正在生成：无字体报错、字号比例完美、上下无缝的高清大图...")

# ==========================================
# 2. 加载数据 
# ==========================================
amir = np.load(os.path.join(data_dir, f'amir_recon_{idx:02d}.npy'))
medl = np.load(os.path.join(data_dir, f'medl_recon_{idx:02d}.npy'))
cmpnet = np.load(os.path.join(data_dir, f'cmpnet_recon_{idx:02d}.npy'))
gt = np.load(os.path.join(data_dir, f'gt_{idx:02d}.npy'))

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

amir, medl, cmpnet, gt = map(normalize, [amir, medl, cmpnet, gt])

# 计算误差图
err_amir = np.abs(amir - gt)
err_medl = np.abs(medl - gt)
err_cmpnet = np.abs(cmpnet - gt)

# ==========================================
# 3. 组织数据
# ==========================================
images = [gt, amir, medl, cmpnet]
errors = [None, err_amir, err_medl, err_cmpnet] 
titles = ['Ground Truth', 'AMIR', 'MEDL-Net', 'CMPNet']
stats = [
    None,                           
    '27.93/0.7513/0.1446',          
    '29.11/0.7604/0.1110',          
    '29.33/0.7650/0.1042'           
]

# ==========================================
# 4. 开始排版画图 (2行 x 4列)
# ==========================================
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# 极限压缩间距！行间距(hspace)为0，原图和误差图完全贴合
plt.subplots_adjust(wspace=0.02, hspace=0)

for i in range(4):
    # ---------------------------------
    # 第一行：重建图 + 局部放大
    # ---------------------------------
    ax_img = axes[0, i]
    ax_img.imshow(images[i], cmap='gray', vmin=0, vmax=1)
    ax_img.axis('off') 

    # ROI 框
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                     linewidth=1.5, edgecolor='lime', facecolor='none')
    ax_img.add_patch(rect)

    # 放大图 (inset)
    axins = ax_img.inset_axes([0.65, 0.02, 0.33, 0.33])
    axins.imshow(images[i][y_min:y_max, x_min:x_max], cmap='gray', vmin=0, vmax=1)
    axins.set_xticks([]) 
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor('lime')
        spine.set_linewidth(2)

    # 红色箭头
    arrow = FancyArrowPatch((0.6, 0.6), (0.4, 0.3), 
                            transform=axins.transAxes,
                            arrowstyle='-|>', mutation_scale=15, 
                            color='red', linewidth=1.5)
    axins.add_patch(arrow)

    # ---------------------------------
    # 第二行：误差图 / 刻度尺 + 标题
    # ---------------------------------
    ax_err = axes[1, i]
    ax_err.axis('off') 

    if i == 0:
        # 第 1 列下方：刻度尺
        norm = Normalize(vmin=0, vmax=0.200)
        sm = ScalarMappable(norm=norm, cmap='jet')
        cax = ax_err.inset_axes([0.47, 0.1, 0.06, 0.8])
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
        cbar.set_ticks([0.000, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200])
        # 🎯 刻度尺数字调整为 12pt，不至于太小
        cbar.ax.tick_params(labelsize=12)
    else:
        # 第 2-4 列下方：误差热力图
        ax_err.imshow(errors[i], cmap='jet', vmin=0, vmax=0.200)
        
        # 白色跑分数字
        if stats[i] is not None:
            text_x = errors[i].shape[1] // 2
            text_y = errors[i].shape[0] - (errors[i].shape[0] * 0.08)
            # 🎯 内部数字提至 14pt
            ax_err.text(text_x, text_y, stats[i], color='white', 
                        fontsize=14, fontweight='bold', ha='center', va='center')

    # 🎯 模型标题提至 18pt，更加醒目大气
    ax_err.text(0.5, -0.12, titles[i], fontsize=18, fontweight='bold', 
                ha='center', va='center', transform=ax_err.transAxes)

save_path = 'Rebuttal_Final_ProFormat.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
print(f"🎉 完美的学术图表已生成: {save_path}")
