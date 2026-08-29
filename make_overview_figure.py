"""
六面板方法总览图 (matplotlib, serif 字体) —— 放 3.7 开头作总览

3 行 x 2 列: CFP | Ground truth / Single-modality | Additive / Weighted | Attention
展示同一 held-out 案例上, 四种方法与真值、原图的整体对比。

字体 serif (Windows 上为 Times New Roman)。去比赛化: 无 "Task" 字样。

用法: 改 IMAGE_ID 和路径后 python make_overview_figure.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import font_manager
from pathlib import Path

def pick_serif():
    for name in ['Times New Roman', 'Times', 'DejaVu Serif']:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return 'serif'

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [pick_serif()]
plt.rcParams['font.size'] = 13

# ============ CONFIG ============
IMAGE_ID = 'g_008'

PANELS = [
    ('CFP (original)',        f'./data/training/images/{IMAGE_ID}.png'),
    ('Ground truth',          f'./data/training/av/{IMAGE_ID}.png'),
    ('Single-modality (CFP)', f'./predictions/task1_train/colored_{IMAGE_ID}.png'),
    ('Additive',              f'./predictions/add_train/colored_{IMAGE_ID}.png'),
    ('Weighted',              f'./predictions/weighted_train/colored_{IMAGE_ID}.png'),
    ('Attention',             f'./predictions/attention_train/colored_{IMAGE_ID}.png'),
]
# ================================


def main():
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    axes = axes.flatten()

    for ax, (label, path) in zip(axes, PANELS):
        if Path(path).exists():
            ax.imshow(mpimg.imread(path))
        else:
            ax.text(0.5, 0.5, f'missing\n{path}', ha='center', va='center', fontsize=9)
        ax.set_title(label, fontsize=14, loc='left', pad=4, fontweight='bold')
        ax.axis('off')

    fig.subplots_adjust(hspace=0.12, wspace=0.06)
    out = f'fig_overview_{IMAGE_ID}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()