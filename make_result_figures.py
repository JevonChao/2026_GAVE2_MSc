"""
生成论文用的结果图 (matplotlib)

产出 4 张 PNG:
  fig_dice.png       —— 四方法 × 三类别 Dice 分组柱状图
  fig_recall.png     —— 四方法 × 三类别 Recall 分组柱状图
  fig_weights.png    —— 学到的模态权重随融合层级变化 (折线)
  fig_biomarker.png  —— 7 个生物标志物的 SMAPE 柱状图

数字直接来自你的 held-out (n=12) 评估结果。
如需改数据,改下方 DATA 区即可。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})

# ==================== DATA ====================
methods = ['Task 1\n(CFP)', 'Add', 'Weighted', 'Attention']
classes = ['Artery', 'Vessel', 'Vein']

dice = {
    'Artery': [0.635, 0.624, 0.673, 0.687],
    'Vessel': [0.757, 0.740, 0.750, 0.757],
    'Vein':   [0.628, 0.717, 0.694, 0.706],
}
recall = {
    'Artery': [0.568, 0.507, 0.616, 0.628],
    'Vessel': [0.727, 0.672, 0.698, 0.716],
    'Vein':   [0.519, 0.657, 0.597, 0.628],
}
# learned modality weights per fusion level
levels = ['L1\n(shallow)', 'L2', 'L3', 'L4\n(deep)']
weights = {
    'FFA_A':  [0.289, 0.291, 0.320, 0.331],
    'FFA_AV': [0.329, 0.307, 0.322, 0.332],
    'CFP':    [0.382, 0.402, 0.358, 0.338],
}
# biomarker SMAPE (%)
bm_names = ['CRAE', 'CRVE', 'AVR', 'Artery\ndensity', 'Vein\ndensity',
            'Artery\nfractal', 'Vein\nfractal']
bm_smape = [7.39, 11.40, 12.46, 37.72, 37.83, 7.97, 11.00]

# greyscale-friendly palette
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
# =============================================


def grouped_bar(metric_dict, title, ylabel, fname, ylim=None):
    x = np.arange(len(classes))
    n = len(methods)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, m in enumerate(methods):
        vals = [metric_dict[c][i] for c in classes]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=m.replace('\n', ' '),
               color=COLORS[i], edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.12),
              frameon=False)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print('saved', fname)


def weights_plot(fname):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(levels))
    markers = {'FFA_A': 'o', 'FFA_AV': '^', 'CFP': 'x'}
    for i, (mod, vals) in enumerate(weights.items()):
        ax.plot(x, vals, marker=markers[mod], color=COLORS[i], linewidth=1.8,
                markersize=7, label=mod, markeredgecolor='black', markeredgewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel('Learned weight (softmax)')
    ax.set_xlabel('Fusion level')
    ax.set_ylim(0.25, 0.45)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print('saved', fname)


def biomarker_plot(fname):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(bm_names))
    colors = ['#4C72B0'] * 3 + ['#C44E52'] * 2 + ['#55A868'] * 2
    bars = ax.bar(x, bm_smape, color=colors, edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(bm_names, fontsize=9)
    ax.set_ylabel('SMAPE (%)')
    for b, v in zip(bars, bm_smape):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f'{v:.1f}',
                ha='center', va='bottom', fontsize=8)
    ax.grid(axis='x', visible=False)
    ax.set_ylim(0, 44)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print('saved', fname)


if __name__ == '__main__':
    grouped_bar(dice, 'Dice', 'Dice coefficient', 'fig_dice.png', ylim=(0.5, 0.80))
    grouped_bar(recall, 'Recall', 'Recall', 'fig_recall.png', ylim=(0.4, 0.80))
    weights_plot('fig_weights.png')
    biomarker_plot('fig_biomarker.png')
    print('\nAll figures generated.')
