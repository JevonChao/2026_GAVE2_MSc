"""
RRWNet 流程图 (matplotlib) —— 按真实递归细化机制绘制

上半: input -> encoder -> decoder -> output(prediction), 输出经递归循环绕回输入
下半: 放大解释 recursive refinement —— original CFP + previous prediction -> refinement -> new prediction

风格与其他图统一 (serif, 柔和配色)。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

BLUE = '#4C72B0'
GREEN = '#55A868'
GREY = '#888888'


def box(ax, x, y, w, h, text, fc='white', ec=BLUE, fs=10, tc='black'):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle='round,pad=0.02,rounding_size=0.06',
                 linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, color=tc, zorder=3)


def arrow(ax, x1, y1, x2, y2, color='black', lw=1.3, rad=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=14, linewidth=lw, color=color, zorder=1,
                 connectionstyle=f'arc3,rad={rad}'))


fig, ax = plt.subplots(figsize=(9, 4.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# ---------- 上半: 主流程 ----------
y0 = 3.5
box(ax, 0.2, y0, 1.5, 0.9, 'CFP\ninput', fc='#EAF0F8', ec=BLUE, fs=9)
box(ax, 2.6, y0, 1.7, 0.9, 'Encoder', fc='white', ec=BLUE, fs=10)
box(ax, 5.0, y0, 1.7, 0.9, 'Decoder', fc='white', ec=BLUE, fs=10)
box(ax, 7.5, y0, 2.1, 0.9, 'Output\n(A / V / vessel)', fc='#EAF0F8', ec=BLUE, fs=9)

arrow(ax, 1.7, y0+0.45, 2.6, y0+0.45)
arrow(ax, 4.3, y0+0.45, 5.0, y0+0.45)
arrow(ax, 6.7, y0+0.45, 7.5, y0+0.45)

# 递归循环: output 底部绕回 encoder 前
ax.add_patch(FancyArrowPatch(
    (8.55, y0), (2.45, y0+0.45),
    arrowstyle='-|>', mutation_scale=14, linewidth=1.3, color=GREY, zorder=1,
    connectionstyle='arc3,rad=0.35'))
ax.text(5.4, 2.55, 'recursive refinement: shared weights, ×5 iterations',
        ha='center', va='center', fontsize=9, color=GREY, style='italic')

# ---------- 下半: 放大解释 refinement ----------
y1 = 0.5
box(ax, 0.2, y1, 2.5, 1.0, 'Original CFP\n+ previous prediction', fc='#EAF3EC', ec=GREEN, fs=9)
box(ax, 3.9, y1, 2.2, 1.0, 'Refinement\nmodule', fc='white', ec=GREEN, fs=10)
box(ax, 7.3, y1, 2.3, 1.0, 'New\nprediction', fc='#EAF3EC', ec=GREEN, fs=9)
arrow(ax, 2.7, y1+0.5, 3.9, y1+0.5, color=GREEN)
arrow(ax, 6.1, y1+0.5, 7.3, y1+0.5, color=GREEN)

# 连接上下: 虚线表示下半是上半循环的展开
ax.annotate('', xy=(1.45, y1+1.0), xytext=(3.4, y0),
            arrowprops=dict(arrowstyle='-', color=GREY, lw=0.8, linestyle=':'))
ax.annotate('', xy=(8.4, y1+1.0), xytext=(8.5, y0),
            arrowprops=dict(arrowstyle='-', color=GREY, lw=0.8, linestyle=':'))

fig.tight_layout()
fig.savefig('fig_pipeline_task1.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('saved fig_pipeline_task1.png')
