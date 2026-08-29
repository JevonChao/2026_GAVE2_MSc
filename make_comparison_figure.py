"""
定性对比图拼接 —— Task1 / Add / Weighted / Attention

把四个方法对同一张 held-out 图的 AV 预测(红蓝绿可视化)横向拼成一张对比图,
每张上方加方法标签,可选在最左侧加原始 CFP 和真值标注,方便直接放进论文。

用法(先按需修改下方 CONFIG 里的路径),然后:
    python make_comparison_figure.py

输出: comparison_<image>.png
"""

import cv2
import numpy as np
from pathlib import Path

# ============ CONFIG: 按你的实际路径修改 ============
IMAGE_ID = 'g_008'                       # 要对比的 held-out 图编号

# 每个方法的 colored 预测图路径(改成你实际的文件夹)
PANELS = [
    ('CFP (original)', f'./data/training/images/{IMAGE_ID}.png',              'raw'),
    ('Ground truth',   f'./data/training/av/{IMAGE_ID}.png',                  'raw'),
    ('Task 1 (CFP)',   f'./predictions/task1_train/colored_{IMAGE_ID}.png',  'pred'),
    ('Add',            f'./predictions/add_train/colored_{IMAGE_ID}.png',     'pred'),
    ('Weighted',       f'./predictions/weighted_train/colored_{IMAGE_ID}.png','pred'),
    ('Attention',      f'./predictions/attention_train/colored_{IMAGE_ID}.png','pred'),
]

# PANEL_W = 480          # 每个子图宽度(等比缩放)
PANEL_W = 620          # 从 480 调大,3x2 布局每格更清晰
LABEL_H = 40           # 标签条高度
GAP = 12               # 子图间距
BG = (255, 255, 255)   # 背景白
# ===================================================


def load_panel(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f'  [warn] cannot read {path}, using blank')
        return np.full((320, PANEL_W, 3), 200, dtype=np.uint8)
    h, w = img.shape[:2]
    new_h = int(h * PANEL_W / w)
    return cv2.resize(img, (PANEL_W, new_h), interpolation=cv2.INTER_AREA)


def add_label(panel, text):
    h, w = panel.shape[:2]
    label = np.full((LABEL_H, w, 3), 255, dtype=np.uint8)
    cv2.putText(label, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([label, panel])


def main():
    panels = []
    for label, path, _kind in PANELS:
        if not Path(path).exists():
            print(f'  [skip] {label}: {path} not found')
            continue
        p = load_panel(path)
        p = add_label(p, label)
        panels.append(p)

    if not panels:
        print('No panels found. Check the paths in CONFIG.')
        return

    # 统一每个 panel 到相同宽高(顶部对齐,底部补白)
    max_h = max(p.shape[0] for p in panels)
    max_w = max(p.shape[1] for p in panels)
    fixed = []
    for p in panels:
        pad_b = max_h - p.shape[0]
        pad_r = max_w - p.shape[1]
        if pad_b > 0:
            p = np.vstack([p, np.full((pad_b, p.shape[1], 3), 255, dtype=np.uint8)])
        if pad_r > 0:
            p = np.hstack([p, np.full((p.shape[0], pad_r, 3), 255, dtype=np.uint8)])
        fixed.append(p)

    # 3 行 x 2 列
    n_cols = 2
    gap_v = np.full((max_h, GAP, 3), 255, dtype=np.uint8)   # 列间距
    rows = []
    for i in range(0, len(fixed), n_cols):
        group = fixed[i:i+n_cols]
        while len(group) < n_cols:
            group.append(np.full((max_h, max_w, 3), 255, dtype=np.uint8))
        row = group[0]
        for p in group[1:]:
            row = np.hstack([row, gap_v, p])
        rows.append(row)

    # 行间距
    row_w = rows[0].shape[1]
    gap_h = np.full((GAP, row_w, 3), 255, dtype=np.uint8)
    grid = rows[0]
    for r in rows[1:]:
        grid = np.vstack([grid, gap_h, r])

    out = f'comparison_{IMAGE_ID}.png'
    cv2.imwrite(out, grid)
    print(f'\nSaved: {out}  ({grid.shape[1]}x{grid.shape[0]})')




if __name__ == '__main__':
    main()
