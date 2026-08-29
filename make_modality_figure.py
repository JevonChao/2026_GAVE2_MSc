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

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

IMAGE_ID = 'g_002'

PANELS = [
    ('Colour fundus (CFP)',      f'./data/training/images/{IMAGE_ID}.png',  'color'),
    ('FFA arterial phase',       f'./data/training/FFA_A/{IMAGE_ID}.png',   'gray'),
    ('FFA arteriovenous phase',  f'./data/training/FFA_AV/{IMAGE_ID}.png',  'gray'),
    ('Artery/vein annotation',   f'./data/training/av/{IMAGE_ID}.png',      'color'),
]

def main():
    n = len(PANELS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4))
    for ax, (label, path, kind) in zip(axes, PANELS):
        if not Path(path).exists():
            ax.text(0.5, 0.5, 'missing', ha='center', va='center')
            ax.set_title(label, fontsize=12)
            ax.axis('off')
            print(f'  [skip] {label}: {path} not found')
            continue
        if kind == 'gray':
            img = mpimg.imread(path)
            if img.ndim == 3:
                img = img[:, :, 0]
            # 判断灰度范围: matplotlib读PNG可能是0-1或0-255
            vmax = 1.0 if img.max() <= 1.0 else 255
            ax.imshow(img, cmap='gray', vmin=0, vmax=vmax)
        else:
            ax.imshow(mpimg.imread(path))
        ax.set_title(label, fontsize=12, pad=6)
        ax.axis('off')
    fig.tight_layout()
    out = f'fig_modality_{IMAGE_ID}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out}')

if __name__ == '__main__':
    main()