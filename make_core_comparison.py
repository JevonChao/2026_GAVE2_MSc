"""
核心定性对比图 (带放大框) —— 模仿 RRWNet 论文风格

布局: 三列并排, 每列 = 主图(带彩色框) + 下方一条放大条(该列的各框裁剪并排)。
放大条的总宽度严格等于主图宽度, 各放大块高度一致, 因此三列完全对齐。

同时把中间产物分开保存:
  1) 带彩色标注框的主图      -> <OUT_DIR>/boxed/
  2) 带同色边框的放大裁剪图  -> <OUT_DIR>/zoom/
  3) 单列成品 (主图+放大条)  -> <OUT_DIR>/column/

坐标系: 框坐标在 task1 预测图 (992x1504) 上量得。
GT (1024x1536) 会先 resize 到预测图尺寸, 使三列对齐。

用法: 改 CASE / BOXES / 路径后 python make_core_comparison.py
"""

import cv2
import numpy as np
from pathlib import Path

# ============ CONFIG ============
CASE = 'g_008'

# 三列: (标签, 文件名短标识, 路径, 是否为GT需要resize)
COLUMNS = [
    ('Ground truth',          'gt',        f'./data/training/av/{CASE}.png',                    True),
    ('Single-modality (CFP)', 'task1',     f'./predictions/task1_train/colored_{CASE}.png',     False),
    ('Attention (ours)',      'attention', f'./predictions/attention_train/colored_{CASE}.png', False),
]

# 目标统一尺寸 (预测图尺寸, 宽 x 高)
TARGET_W, TARGET_H = 1504, 992

# 放大框: (x, y, w, h, 颜色BGR)  —— 坐标在 task1 预测图上量得
BOXES = [
    (1158, 666, 184, 278, (0, 200, 0)),    # 框1 绿
    (289,  7,   162, 250, (255, 100, 0)),  # 框2 蓝
    (649,  686, 267, 151, (0, 220, 220)),  # 框3 黄
]

# 放大条中各框从左到右的排列顺序 (BOXES 的下标)。None = 按 BOXES 原序
ZOOM_ORDER = [1, 2, 0]

DISP_W = 620        # 每列显示宽度 (主图与放大条同宽)
BOX_THICK = 5       # 主图框线粗细
ZOOM_BORDER = 6     # 放大块外框粗细
GAP = 8             # 放大块之间的间距
V_GAP = 10          # 主图与放大条之间的垂直间距
COL_GAP = 26        # 列与列之间的间距
LABEL_H = 44        # 顶部列标签高度 (SHOW_LABELS=False 时忽略)
SHOW_LABELS = False # 是否在每列顶部写标签
BG = (255, 255, 255)

OUT_DIR = Path(f'fig_core_{CASE}')
SAVE_BOXED_FULLRES = True   # 单独保存的带框主图是否用原始分辨率
SAVE_PARTS = True           # 是否保存 boxed / zoom / column 单图
# ================================


def load_col(path, is_gt):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return np.full((TARGET_H, TARGET_W, 3), 200, dtype=np.uint8)
    if is_gt:
        img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
    elif img.shape[1] != TARGET_W or img.shape[0] != TARGET_H:
        img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    return img


def draw_boxes(img):
    out = img.copy()
    for (x, y, w, h, color) in BOXES:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, BOX_THICK)
    return out


def to_disp(im, w=DISP_W):
    h = max(int(round(im.shape[0] * w / im.shape[1])), 1)
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)


def plan_zoom_strip(order):
    """求解统一的放大块内容高度 H, 使得整条放大条宽度恰好等于 DISP_W。

    返回 (H, [每个放大块的内容宽度]).
    """
    n = len(order)
    aspects = [BOXES[i][2] / BOXES[i][3] for i in order]     # w/h
    budget = DISP_W - GAP * (n - 1) - 2 * ZOOM_BORDER * n     # 留给图像内容的总宽
    if budget <= 0:
        raise ValueError('DISP_W 太小, 放不下放大条')
    h = budget / sum(aspects)
    widths = [max(int(round(a * h)), 1) for a in aspects]
    # 四舍五入的误差全部补到最后一块, 保证宽度精确对齐
    widths[-1] += budget - sum(widths)
    return max(int(round(h)), 1), widths


def make_zoom(raw, box, out_w, out_h, border=ZOOM_BORDER):
    x, y, w, h, color = box
    crop = raw[y:y + h, x:x + w].copy()
    z = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(z, border, border, border, border,
                              cv2.BORDER_CONSTANT, value=color)


def hcat(imgs, gap, bg=BG):
    """等高横向拼接。"""
    h = imgs[0].shape[0]
    sep = np.full((h, gap, 3), bg, dtype=np.uint8)
    out = imgs[0]
    for im in imgs[1:]:
        out = np.hstack([out, sep, im])
    return out


def add_top_label(panel, text):
    w = panel.shape[1]
    label = np.full((LABEL_H, w, 3), 255, dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(label, text, (max((w - tw) // 2, 4), 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([label, panel])


def main():
    # 1) 载入三列
    cols_raw = []
    for label, key, path, is_gt in COLUMNS:
        if not Path(path).exists():
            print(f'[skip] {label}: {path} not found')
            return
        cols_raw.append(load_col(path, is_gt))

    order = ZOOM_ORDER if ZOOM_ORDER else list(range(len(BOXES)))
    zoom_h, zoom_ws = plan_zoom_strip(order)

    if SAVE_PARTS:
        for sub in ('boxed', 'zoom', 'column'):
            (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    columns = []
    for raw, (lbl, key, _, _) in zip(cols_raw, COLUMNS):
        # 2) 主图: 画框
        boxed = draw_boxes(raw)
        if SAVE_PARTS:
            save_img = boxed if SAVE_BOXED_FULLRES else to_disp(boxed)
            p = OUT_DIR / 'boxed' / f'{CASE}_boxed_{key}.png'
            cv2.imwrite(str(p), save_img)
            print(f'  [boxed ] {p}  ({save_img.shape[1]}x{save_img.shape[0]})')
        main_disp = to_disp(boxed)

        # 3) 放大块: 统一高度, 宽度按各框比例
        zooms = []
        for slot, bi in enumerate(order):
            z = make_zoom(raw, BOXES[bi], zoom_ws[slot], zoom_h)
            zooms.append(z)
            if SAVE_PARTS:
                p = OUT_DIR / 'zoom' / f'{CASE}_box{bi + 1}_{key}.png'
                cv2.imwrite(str(p), z)
                print(f'  [zoom  ] {p}  ({z.shape[1]}x{z.shape[0]})')
        strip = hcat(zooms, GAP)

        # 4) 组列: 主图 + 垂直间距 + 放大条
        assert strip.shape[1] == main_disp.shape[1], \
            f'放大条宽 {strip.shape[1]} != 主图宽 {main_disp.shape[1]}'
        vgap = np.full((V_GAP, DISP_W, 3), BG, dtype=np.uint8)
        col = np.vstack([main_disp, vgap, strip])
        if SHOW_LABELS:
            col = add_top_label(col, lbl)
        columns.append(col)

        if SAVE_PARTS:
            p = OUT_DIR / 'column' / f'{CASE}_column_{key}.png'
            cv2.imwrite(str(p), col)
            print(f'  [column] {p}  ({col.shape[1]}x{col.shape[0]})')

    # 5) 三列并排 (高度天然一致)
    final = hcat(columns, COL_GAP)
    out = f'fig_core_comparison_{CASE}.png'
    cv2.imwrite(out, final)
    print(f'Saved: {out}  ({final.shape[1]}x{final.shape[0]})')


if __name__ == '__main__':
    main()