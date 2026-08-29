"""
泛化定性展示图 (g_051-g_100, 无标注)

对无标注数据展示 CFP 输入 + attention 模型预测,
说明模型在未见、无标注数据上仍能输出合理、拓扑连贯的分割。
无真值,故不做指标对比,仅作定性泛化展示。

布局: 4 行 (4个案例) × 2 列 (CFP | prediction), serif 字体标签。

用法: 改 CASES 为你挑选的 4 个案例编号, 然后 python make_generalisation_figure.py
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
plt.rcParams['font.size'] = 12

# ============ CONFIG: 改成你挑选的 4 个案例 ============
CASES = ['g_066', 'g_070']

CFP_DIR = './data/validation/images'
PRED_DIR = './predictions/task2_attention'   # colored_g_0XX.png
# =====================================================


def main():
    n = len(CASES)
    fig, axes = plt.subplots(n, 2, figsize=(7.5, 3.0 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for row, case in enumerate(CASES):
        cfp = Path(CFP_DIR) / f'{case}.png'
        pred = Path(PRED_DIR) / f'colored_{case}.png'

        # CFP
        ax = axes[row, 0]
        if cfp.exists():
            ax.imshow(mpimg.imread(cfp))
        else:
            ax.text(0.5, 0.5, f'missing\n{cfp}', ha='center', va='center', fontsize=8)
        ax.axis('off')
        if row == 0:
            ax.set_title('Colour fundus (CFP)', fontsize=12, pad=6)

        # prediction
        ax = axes[row, 1]
        if pred.exists():
            ax.imshow(mpimg.imread(pred))
        else:
            ax.text(0.5, 0.5, f'missing\n{pred}', ha='center', va='center', fontsize=8)
        ax.axis('off')
        if row == 0:
            ax.set_title('Predicted artery/vein (attention)', fontsize=12, pad=6)

        # 左侧行标签 (案例名)
        axes[row, 0].text(-0.06, 0.5, case, transform=axes[row, 0].transAxes,
                          rotation=90, ha='center', va='center', fontsize=11)

    fig.tight_layout()
    out = 'fig_generalisation.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
